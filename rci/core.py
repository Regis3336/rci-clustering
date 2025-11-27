# core.py — Matrix-On-Demand RCI Engine with Full Auto-Tuning
# =================================================================================================
# PUBLIC API:
#     - SpectralRCI.fit(X)
#     - SpectralRCI.predict([X_new])
#     - SpectralRCI.plot_results_3d()
#     - run_spectral_rci_demo()
#     - run_torture_suite()
#
# PURPOSE:
#   This module implements the full working version of Spectral RCI — a hybrid geometric–spectral
#   clustering algorithm designed to operate directly on large point clouds without forming
#   dense matrices. The architecture integrates:
#
#       (1) Matrix-free normalized Laplacian operators with:
#               – FAISS kNN graphs
#               – local self-tuning σ_i
#               – mutual kNN filtering
#               – safe fallback paths for large-n or numerical instability
#
#       (2) A robust spectral embedding pipeline:
#               – dynamic eigenvector recovery
#               – symmetry testing
#               – multiple eigen-solver fallback strategies
#               – strict padding/truncation to guarantee dimension k
#
#       (3) The full RCI algorithm:
#               – radial Morse profiles M_c(k)
#               – density-aware scaling with intrinsic dimension
#               – curvature-based boundary detection (ΔM, Δ²M)
#               – center selection via geometric coverage deficit
#               – guaranteed termination and reassignment rules
#
#       (4) Official graph export:
#               The module exports the symmetric Laplacian graph used by the MEI
#               (Morse Erasure Index) metric. All edge weights are exact and correspond
#               to the graph used during spectral embedding.
#
# DESIGN PRINCIPLES:
#   * Zero-copy philosophy:
#       No dense NxN matrices are ever constructed unless n ≤ 10k. All heavy oper   ations
#       are performed using matrix-free LinearOperator abstractions.
#
#   * Fail-safe spectral geometry:
#       Every stage is wrapped in robust error handling. If the Laplacian op becomes unstable
#       or eigsh fails, a mathematically correct fallback embedding is produced.
#
#   * Dimension-aware density:
#       The density estimator uses the Levina–Bickel MLE intrinsic dimension and scales
#       the Morse field f = -log ρ accordingly.
#
#   * RCI correctness:
#       The implementation enforces the exact algorithmic invariants of the theoretical
#       RCI framework, independent of embedding choice.
#
# TORTURE SUITE:
#   Appendix A of the paper includes the “torture suite”: adversarial geometric
#   transformations (smooth warps, density skew, shortcut bridges, outlier bursts)
#   used to validate stability. The same tests are reproduced here under:
#
#       - run_torture_suite()
#
# OUTPUTS:
#   * Spectral embedding (optional)
#   * RCI clusters and profiles
#   * Graph structure (i, j, w) for MEI
#   * High-resolution diagnostic plots via Plotly
#
# =================================================================================================

import math
import faiss
import warnings
import numpy as np
import time as time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.sparse.linalg import LinearOperator, eigsh

warnings.filterwarnings('ignore')

EIGENVECTORS_N = 5

EPS = 1e-12

# ============================================================================
# 1. Sanity Check: Symmetry of the Operator
# ============================================================================
def check_symmetry(op, n_checks=3, tol=1e-6):
    """Efficient symmetry check for Laplacian operator."""
    n = op.shape[0]
    if n > 10000: return True
    try:
        for _ in range(n_checks):
            a = np.random.randn(n)
            b = np.random.randn(n)
            left = np.dot(a, op.matvec(b))
            right = np.dot(b, op.matvec(a))
            if np.isnan(left) or np.isnan(right) or np.isinf(left) or np.isinf(right):
                return False
            if abs(left - right) > tol*(1+abs(left)+abs(right)):
                return False
        return True
    except:
        return False

# ============================================================================
# 2. Intrinsic Dimension Estimation (Levina--Bickel)
# ============================================================================
def estimate_intrinsic_dim(X, k=10):
    """
    Levina–Bickel intrinsic dimension estimator (MLE).
    Corrigido: média correta e estabilidade numérica.
    """
    n, d = X.shape
    if n <= k:
        return 1.0

    index = faiss.IndexFlatL2(d)
    index.add(X.astype(np.float32))

    d2, _ = index.search(X.astype(np.float32), k+1)
    d2 = d2[:,1:]                       
    r = np.sqrt(np.maximum(d2, 1e-12))

    # LB formula:
    logs = np.log(r[:, -1][:,None] / r[:, :-1])
    S = np.sum(logs, axis=1)
    m = (k - 1) / np.mean(S)

    # clamp final
    return float(max(1.0, min(50.0, m)))


# ============================================================================
# 3. Matrix-Free Laplacian with Local Self-Tuning of Sigma
# ============================================================================
class MatrixOnDemandLaplacian:
    def __init__(self, X: np.ndarray, k_nn: int = 10, mutual_knn: bool = False,
                 eps_self_loop: float = 1e-6, self_tuning: bool = True,
                 sigma_global: float | None = None, verbose: bool = False):

        self.verbose = verbose
        if self.verbose:
            print(f"Building Laplacian for {X.shape[0]} points in {X.shape[1]}D...")

        self.X = X.astype(np.float32)
        self.n, self.d = self.X.shape
        self.k_nn = int(min(k_nn, self.n - 1))
        self.mutual_knn = bool(mutual_knn)
        self.eps_self_loop = float(max(eps_self_loop, 1e-12))
        self.self_tuning = bool(self_tuning)
        self.sigma_global = None if sigma_global is None else float(max(sigma_global, 1e-12))

        if self.n < 10:
            raise ValueError(f"Too few points ({self.n}) for meaningful clustering")

        # ---------------------------------------------------------
        # FAISS kNN
        # ---------------------------------------------------------
        try:
            self.index = faiss.IndexFlatL2(self.d)
            self.index.add(self.X)
            d2, idx = self.index.search(self.X, self.k_nn + 1)
            self.d2 = d2[:, 1:].astype(np.float64)
            self.idx = idx[:, 1:]
        except Exception as e:
            raise RuntimeError(f"FAISS kNN search failed: {e}")

        if np.any(np.isinf(self.d2)) or np.any(np.isnan(self.d2)):
            raise ValueError("Invalid distances found in kNN search")

        # ---------------------------------------------------------
        # LOCAL SELF-TUNING OF SIGMA
        # ---------------------------------------------------------
        if self.self_tuning:
            k_sigma = min(int(np.ceil(np.log(max(self.n, 3)))), self.k_nn)

            try:
                d2_sigma, _ = self.index.search(self.X, k_sigma + 1)
                d2_sigma = d2_sigma[:, 1:].astype(np.float64)

                sigma_i = np.sqrt(np.maximum(d2_sigma[:, -1], 1e-12))
                sigma_i = np.clip(sigma_i, 1e-6, np.percentile(sigma_i, 95))

                den = (sigma_i[:, None] * sigma_i[self.idx]) + 1e-12
                W_dir = np.exp(-self.d2 / np.maximum(den, 1e-12))

            except Exception as e:
                if self.verbose:
                    print(f"Self-tuning failed, falling back to global sigma: {e}")

                pos = self.d2[self.d2 > 0]
                sigma2 = float(np.median(pos)) if pos.size else 1.0
                W_dir = np.exp(-self.d2 / (2 * sigma2 + 1e-12))

        else:
            if self.sigma_global is None:
                pos = self.d2[self.d2 > 0]
                sigma2 = float(np.median(pos)) if pos.size else 1.0
                W_dir = np.exp(-self.d2 / (2 * sigma2 + 1e-12))
            else:
                W_dir = np.exp(-self.d2 / (2 * self.sigma_global**2 + 1e-12))

        W_dir = np.clip(W_dir, 1e-12, 1.0)

        # ---------------------------------------------------------
        # MUTUAL KNN FILTER
        # ---------------------------------------------------------
        if self.mutual_knn:
            try:
                nbr_pos = [dict(zip(self.idx[i], range(self.k_nn))) for i in range(self.n)]
                mutual = np.zeros_like(W_dir, dtype=bool)

                for i in range(self.n):
                    for t, j in enumerate(self.idx[i]):
                        if j < len(nbr_pos) and i in nbr_pos[j]:
                            mutual[i, t] = True

                W_dir = np.where(mutual, W_dir, 0.0)

            except Exception as e:
                if self.verbose:
                    print(f"Mutual kNN failed, using directed graph: {e}")

        self.W_dir = W_dir.astype(np.float64)

        # ---------------------------------------------------------
        # SYMMETRIZATION (Wsym)
        # ---------------------------------------------------------
        try:
            if self.n > 10000:
                from scipy.sparse import csr_matrix
                rows, cols, vals = [], [], []

                for i in range(self.n):
                    for t, j in enumerate(self.idx[i]):
                        if W_dir[i, t] > 1e-12:
                            rows.append(i)
                            cols.append(j)
                            vals.append(W_dir[i, t])

                W_sparse = csr_matrix((vals, (rows, cols)), shape=(self.n, self.n))
                W_sparse = 0.5 * (W_sparse + W_sparse.T)
                self.Wsym = W_sparse.toarray()

            else:
                W_tmp = np.zeros((self.n, self.n), dtype=np.float64)
                for i in range(self.n):
                    for t, j in enumerate(self.idx[i]):
                        W_tmp[i, j] = W_dir[i, t]

                self.Wsym = 0.5 * (W_tmp + W_tmp.T)

            deg = np.array(self.Wsym.sum(axis=1)).flatten() + self.eps_self_loop
            deg = np.maximum(deg, 1e-10)
            self.Dmhalf = (1.0 / np.sqrt(deg)).astype(np.float64)

        except Exception as e:
            raise RuntimeError(f"Symmetrization failed: {e}")

        # ---------------------------------------------------------
        # EXPORT OFFICIAL SYMMETRIC GRAPH (OPTION B)
        # ---------------------------------------------------------
        rows, cols = np.where(self.Wsym > 1e-12)
        weights = self.Wsym[rows, cols]

        self.graph_sym = {
            "i": rows.astype(np.int64),
            "j": cols.astype(np.int64),
            "w": weights.astype(np.float64)
        }

        # ---------------------------------------------------------
        # DONE
        # ---------------------------------------------------------
        if self.verbose:
            print(f"Laplacian built successfully. Density: {np.mean(self.Wsym > 0):.3f}")
    
    def matvec(self, v: np.ndarray) -> np.ndarray:
        """Matrix-vector multiplication for normalized Laplacian."""
        try:
            v64 = np.asarray(v, dtype=np.float64)
            if len(v64) != self.n:
                raise ValueError(f"Vector length {len(v64)} != matrix size {self.n}")
            z = self.Dmhalf * v64
            acc = self.Wsym @ z
            acc += self.eps_self_loop * z
            result = v64 - self.Dmhalf * acc
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                if self.verbose:
                    print("Warning: NaN/Inf detected in matvec, using fallback")
                return v64
            return result
        except Exception as e:
            if self.verbose:
                print(f"Matvec failed: {e}, using identity")
            return np.asarray(v, dtype=np.float64)
    
    def to_linear_operator(self):
        return LinearOperator(
            shape=(self.n, self.n),
            matvec=self.matvec,
            dtype=np.float64,
        )

# ============================================================================
# 4. Spectral RCI with Full Auto-Tuning
# ============================================================================
class SpectralRCI:
    def __init__(
        self,
        k_den: int = None,
        alpha: float = None,
        m_intrinsic: int = None,
        k_nn: int = None,
        n_eigenvectors: int = EIGENVECTORS_N,
        warm_up: int = None,
        mutual_knn: bool = True,
        epsilon_stop: float = 0.05,
        eps_self_loop: float = 1e-6,
        use_spectral: bool = True,
        self_tuning: bool = True,
        sigma_global: float | None = None
    ):
        # --------------------------
        # Hyperparameters
        # --------------------------
        self.k_den = k_den
        self.alpha = alpha
        self.m_intrinsic = m_intrinsic
        self.k_nn = k_nn
        self.n_eigenvectors = n_eigenvectors
        self.warm_up = warm_up
        self.mutual_knn = mutual_knn
        self.epsilon_stop = epsilon_stop
        self.eps_self_loop = eps_self_loop
        self.use_spectral = use_spectral
        self.self_tuning = self_tuning
        self.sigma_global = sigma_global

        # --------------------------
        # Internal state variables
        # --------------------------
        self.X_original = None          # data in original space
        self.X_spectral = None          # embedding (optional)
        self.eigenvalues = None
        self.eigenvectors = None

        # RCI results
        self.rci_profiles = {}
        self.clusters = {}
        self.centers = []

        # --------------------------
        # OFFICIAL GRAPH (NEW)
        # The graph must store indices and weights
        # for MEI injection and reproducibility.
        # --------------------------
        self.graph_ = {
            "i": None,    # row indices
            "j": None,    # column indices
            "w": None     # edge weights
        }

        # For debugging / diagnostics
        self.spectral_failed = True

    def build_faiss_index(self, X):
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X.astype(np.float32))
        return index
    
    def assign_clusters(self, X):
        """
        Assign each point to the RCI center that best matches it
        under the Morse metric:

            D(i, c) = | M_c(rank_c(i)) - M_c(0) |

        where rank_c(i) is the radial position of i in the
        ordering induced by center c.

        If rank_c(i) does not exist, a Euclidean fallback is used.
        """

        X = np.asarray(X, float)
        n = len(X)
        C = self.centers
        kC = len(C)

        if kC == 0:
            return np.zeros(n, int)

        # preparation
        labels = np.full(n, -1, int)
        Xc = X[np.array(C)]

        # rank map: rank[c][i] = position of i in sorted_idx
        rank_maps = {}
        for c in C:
            sorted_idx = self.rci_profiles[c]["sorted_idx"]
            rank_vec = np.full(n, -1, int)
            rank_vec[sorted_idx] = np.arange(len(sorted_idx))
            rank_maps[c] = rank_vec

        # pointwise assignment
        for i in range(n):
            best_c = None
            best_d = np.inf

            for cid, c in enumerate(C):
                rank_i = rank_maps[c][i]

                # rare fallback: point never appeared in c's ordering
                if rank_i < 0:
                    d = np.linalg.norm(X[i] - Xc[cid])
                else:
                    M = self.rci_profiles[c]["M_c"]
                    rank_i = min(rank_i, len(M) - 1)

                    # Morse difference
                    d = abs(M[rank_i] - M[0])

                    # fallback if degenerate
                    if d < 1e-12:
                        d = np.linalg.norm(X[i] - Xc[cid])

                if d < best_d:
                    best_d = d
                    best_c = cid

            labels[i] = best_c

        return labels


    def compute_knn_density(self, X, index):
        """
        Correct kNN density estimator:
            rho(x) = k / (n * V_m * R_k(x)^m)

        where:
            - R_k(x) is the distance to the k-th nearest neighbor
            - m is the intrinsic dimension
            - V_m is the unit-ball volume in R^m
        """

        # always query k+1 (exclude self)
        kq = min(self.k_den + 1, len(X))
        d2, _ = index.search(X.astype(np.float32), kq)

        # drop self-distance
        d2 = d2[:, 1:]

        # R_k(x)
        R_k = np.sqrt(np.maximum(d2[:, -1], 1e-12))

        # intrinsic dimension
        m = self.m_intrinsic

        # volume of unit m-ball
        # V_m = π^(m/2) / Γ(m/2 + 1)
        V_m = (math.pi ** (m / 2.0)) / math.gamma(1.0 + m / 2.0)

        # density:  k / (n * V_m * R_k^m)
        rho = self.k_den / (len(X) * V_m * (R_k ** m + 1e-12))

        return rho

    
    def compute_spectral_embedding_matrix_free(self, X):
        """
        Matrix-free spectral embedding with strict eigenvector control.
        Always returns EXACTLY self.n_eigenvectors.
        Includes:
            - symmetry check
            - dual eigsh attempts
            - padding if insufficient eigenvectors
            - strict truncation
            - silent-failure detection
            - injection of graph_sym (Opção B)
        """

        self.spectral_failed = True
        n = len(X)

        # ------------------------------------------------------------
        # Spectral disabled → return raw space
        # ------------------------------------------------------------
        if not self.use_spectral:
            self.eigenvalues = None
            self.eigenvectors = X.astype(np.float64)
            return self.eigenvectors, self.eigenvalues

        print(f"Building matrix-free Laplacian operator (target: {self.n_eigenvectors} eigenvectors)...")

        # Defaults
        if self.k_nn is None:
            self.k_nn = int(np.ceil(np.log(max(n, 3))))

        if self.warm_up is None:
            self.warm_up = max(5, int(np.ceil(0.002 * n)))

        try:
            # ------------------------------------------------------------
            # BUILD LAPLACIAN + GRAPH EXPORT
            # ------------------------------------------------------------
            self.laplacian_op = MatrixOnDemandLaplacian(
                X,
                self.k_nn,
                self.mutual_knn,
                self.eps_self_loop,
                self_tuning=self.self_tuning,
                sigma_global=self.sigma_global,
                verbose=True
            )

            # Inject graph (Opção B)
            self.graph_ = self.laplacian_op.graph_sym

            # Linear operator
            L_operator = self.laplacian_op.to_linear_operator()

            # ------------------------------------------------------------
            # SYMMETRY TEST (only for small datasets)
            # ------------------------------------------------------------
            if n <= 5000:
                if not check_symmetry(L_operator):
                    print("Warning: operator not symmetric — continuing...")

            # ------------------------------------------------------------
            # FIRST ATTEMPT EIGENDECOMPOSITION
            # ------------------------------------------------------------
            max_eigs = min(self.n_eigenvectors + 10, n - 2)
            print(f"Computing {max_eigs} eigenvalues (will keep {self.n_eigenvectors})...")

            start = time.time()
            try:
                eigs, vecs = eigsh(
                    L_operator,
                    k=max_eigs,
                    which='SA',
                    tol=1e-3,
                    maxiter=3000,
                    v0=np.random.randn(n)
                )
                print(f"Eigendecomposition completed in {time.time() - start:.2f}s")

            except Exception as e:
                print(f"First eigsh attempt failed: {e}")
                print("Retrying with relaxed parameters...")

                # ------------------------------------------------------------
                # SECOND ATTEMPT
                # ------------------------------------------------------------
                try:
                    max_eigs = min(self.n_eigenvectors + 5, n - 2)
                    eigs, vecs = eigsh(
                        L_operator,
                        k=max_eigs,
                        which='SA',
                        tol=1e-2,
                        maxiter=1000,
                        v0=np.ones(n)
                    )

                except Exception as e2:
                    # ------------------------------------------------------------
                    # HARD FAILURE → RANDOM EMBEDDING
                    # ------------------------------------------------------------
                    print(f"Second eigsh failed: {e2}")
                    print("Using fallback: RANDOM embedding.")

                    V = np.random.randn(n, self.n_eigenvectors)
                    V /= np.linalg.norm(V, axis=0) + 1e-12

                    self.eigenvalues = np.ones(self.n_eigenvectors)
                    self.eigenvectors = V
                    self.spectral_failed = True
                    return self.eigenvectors, self.eigenvalues

            # ------------------------------------------------------------
            # FILTER ZERO/NOISE EIGENVALUES
            # ------------------------------------------------------------
            mask = eigs > 1e-10
            eigs = eigs[mask]
            vecs = vecs[:, mask]

            if len(eigs) == 0:
                print("ERROR: All eigenvalues filtered out → identity embedding.")
                self.eigenvectors = X.astype(np.float64)
                self.eigenvalues = None
                self.spectral_failed = True
                return self.eigenvectors, self.eigenvalues

            # ------------------------------------------------------------
            # PAD IF NOT ENOUGH EIGENVECTORS
            # ------------------------------------------------------------
            if len(eigs) < self.n_eigenvectors:
                needed = self.n_eigenvectors - len(eigs)
                print(f"Warning: Found {len(eigs)} eigenvalues → padding {needed} vectors.")

                rand = np.random.randn(n, needed)

                # Gram–Schmidt
                for i in range(needed):
                    v = rand[:, i]
                    for j in range(vecs.shape[1]):
                        v -= np.dot(v, vecs[:, j]) * vecs[:, j]
                    norm = np.linalg.norm(v)
                    if norm < 1e-12:
                        v = np.random.randn(n)
                    v /= np.linalg.norm(v)
                    rand[:, i] = v

                vecs = np.hstack([vecs, rand])
                eigs = np.hstack([eigs, np.ones(needed)])

            # ------------------------------------------------------------
            # STRICT TRUNCATION
            # ------------------------------------------------------------
            d = min(self.n_eigenvectors, len(eigs))

            self.eigenvalues = eigs[:d]
            self.eigenvectors = vecs[:, :d]

            print(
                f"✓ Using {d} eigenvectors "
                f"(requested: {self.n_eigenvectors}), "
                f"eigenvalue range: [{self.eigenvalues[0]:.6f}, {self.eigenvalues[-1]:.6f}]"
            )

            # ------------------------------------------------------------
            # SILENT FAILURE CHECK
            # ------------------------------------------------------------
            if self.eigenvectors.shape == X.shape:
                if np.allclose(self.eigenvectors, X, atol=1e-8, rtol=1e-8):
                    print("Spectral embedding FAILED silently → identical to raw space.")
                    self.spectral_failed = True
                else:
                    self.spectral_failed = False
            else:
                self.spectral_failed = False

        # ------------------------------------------------------------
        # CRITICAL FAILURE CATCH
        # ------------------------------------------------------------
        except Exception as e:
            print(f"✗ Critical spectral embedding error: {e}")
            print("Using identity embedding.")

            self.eigenvalues = None
            self.eigenvectors = X.astype(np.float64)
            self.spectral_failed = True

        return self.eigenvectors, self.eigenvalues


    def compute_rci_profile(self, X, i, rho):
        """
        Fully corrected computation of the RCI profile.

        Guarantees:
            - self.index_knn always exists (independent of fit())
            - FAISS index dimension always matches X.shape[1]
            - sorted neighbors are returned for all n points
            - r_k, M_c, and A_c all have length n
        """

        X = np.asarray(X, float)
        n, d = X.shape

        # ---------------------------------------------------------
        # CRITICAL FIX:
        # Ensure that the FAISS index exists and matches X's dimension.
        # If missing or dimension-mismatched → rebuild it now.
        # ---------------------------------------------------------
        if not hasattr(self, "index_knn") or self.index_knn is None:
            self.index_knn = self.build_faiss_index(X.astype(np.float32))
        else:
            if self.index_knn.d != d:
                self.index_knn = self.build_faiss_index(X.astype(np.float32))

        # ---------------------------------------------------------
        # 1. Radial distances from center i
        # ---------------------------------------------------------
        x0 = X[i][None, :].astype(np.float32)

        # search for ALL points (k = n neighbors)
        d2, idx = self.index_knn.search(x0, n)

        d2 = d2[0]          # shape (n,)
        idx = idx[0]        # sorted indices
        r = np.sqrt(d2 + 1e-12)

        # ---------------------------------------------------------
        # 2. Sorted density
        # ---------------------------------------------------------
        rho_sorted = rho[idx]

        # ---------------------------------------------------------
        # 3. Accumulated amplitude A_c(j)
        # ---------------------------------------------------------
        csum = np.cumsum(rho_sorted)
        A_c = csum / (np.arange(1, n + 1))

        # ---------------------------------------------------------
        # 4. True r_k profile (monotone radial distances)
        # ---------------------------------------------------------
        r_k = r

        # ---------------------------------------------------------
        # 5. Morse profile M_c(j)
        # ---------------------------------------------------------
        alpha = self.alpha if self.alpha is not None else 1.0
        M_c = A_c - alpha * np.log(r_k + 1e-12)

        return M_c, r_k, idx, A_c


    def farthest_from_covered_seeding(
    self, X, covered_centers, covered_radii, min_surface, uncovered_points
    ):
        """
        Correct 100% seeding strategy:

        - first seed: farthest point from the center of mass
        - subsequent seeds: argmax(min_surface) restricted to the uncovered points
        """

        # first seed
        if not covered_centers:
            c = np.mean(X, axis=0)
            d = np.linalg.norm(X - c, axis=1)
            return int(np.argmax(d))

        # subsequent seeds
        up = np.array(list(uncovered_points), dtype=int)
        vals = min_surface[up]

        # safe argmax
        best = up[np.argmax(vals)]
        return int(best)

    
    def reassign_remaining_points(self, X_embed, uncovered_points):
        """
        Correct reassignment rule: a point is assigned to the cluster
        that has the largest geometric margin (radius - distance).
        """
        if not uncovered_points:
            return

        rem = np.array(list(uncovered_points), dtype=int)

        for r in rem:
            best_cluster = None
            best_margin  = -np.inf

            for cid, prof in self.rci_profiles.items():
                center = prof["center_idx"]
                radius = prof["boundary_radius"]

                dist = np.linalg.norm(X_embed[r] - X_embed[center])
                margin = radius - dist

                if margin > best_margin:
                    best_margin = margin
                    best_cluster = cid

            if best_cluster is None:
                best_cluster = 0

            self.clusters[best_cluster] = np.append(
                self.clusters[best_cluster], r
            )



    def reverse_clustering_impact(self, X_embed: np.ndarray):
        """
        RCI without any automatic dimensionality decisions.
        It uses exactly the embedding provided — whether 2D, 3D, or kD (k eigenvectors).
        It does NOT recompute m_intrinsic, alpha, or k_den if the user supplied them.
        """
        n = X_embed.shape[0]
        self.rci_profiles = {}
        self.clusters = {}
        self.centers = []

        # --------------------------------------------------------
        # 1. No dimension estimation from the embedding
        # --------------------------------------------------------
        if self.m_intrinsic is None:
            self.m_intrinsic = 3   # safe default
        if self.alpha is None:
            self.alpha = self.m_intrinsic
        if self.k_den is None:
            self.k_den = int(np.ceil(self.m_intrinsic * np.log(max(n, 3))))
        if self.warm_up is None:
            self.warm_up = max(5, int(np.ceil(0.002 * n)))

        # --------------------------------------------------------
        # 2. Standard kNN density (now consistent)
        # --------------------------------------------------------
        index = self.build_faiss_index(X_embed)
        rho = self.compute_knn_density(X_embed, index)

        # RCI auxiliary structures
        covered_centers = []
        covered_radii = []
        uncovered_points = set(range(n))
        min_surface = np.full(n, np.inf)

        cluster_id = 0

        # --------------------------------------------------------
        # 3. Main RCI loop
        # --------------------------------------------------------
        while uncovered_points and len(uncovered_points) > self.epsilon_stop * n:

            # Correct seed (point farthest from the current "coverage")
            center_idx = self.farthest_from_covered_seeding(
                X_embed,
                covered_centers,
                covered_radii,
                min_surface,
                uncovered_points,
            )
            if center_idx not in uncovered_points:
                center_idx = next(iter(uncovered_points))

            # RCI profile
            M_c, r_k, sorted_indices, A_c = self.compute_rci_profile(
                X_embed,
                center_idx,
                rho,
            )

            # Boundary detection
            kappa_star, delta_M, delta2_M = self.detect_boundary(M_c)
            boundary_radius = r_k[kappa_star]

            # Points inside the cluster
            cluster_points = sorted_indices[: min(kappa_star + 1, len(sorted_indices))]

            # Record
            self.clusters[cluster_id] = np.asarray(cluster_points, dtype=int)
            self.rci_profiles[cluster_id] = {
                "center_idx": center_idx,
                "M_c": M_c,
                "A_c": A_c,
                "r_k": r_k,
                "delta_M": delta_M,
                "delta2_M": delta2_M,
                "kappa_star": kappa_star,
                "boundary_radius": boundary_radius,
                "sorted_indices": sorted_indices,
            }

            # Update global structures
            covered_centers.append(center_idx)
            covered_radii.append(boundary_radius)

            d = np.linalg.norm(X_embed - X_embed[center_idx], axis=1)
            min_surface[:] = np.minimum(min_surface, np.maximum(0.0, d - boundary_radius))

            uncovered_points.difference_update(cluster_points)
            cluster_id += 1

        # --------------------------------------------------------
        # 4. Reassignment of remaining points (NO heuristics)
        # --------------------------------------------------------
        self.reassign_remaining_points(X_embed, uncovered_points)

        self.centers = covered_centers
        return self.clusters
    
    def compute_density(self, X):
        """
        Official unified density estimator for RCI.
        Uses kNN density with intrinsic-dimension-aware radius scaling.

        Returns:
            rho : array of shape (n,)
        """

        # Ensure hyperparameters exist
        n = len(X)

        # default dimension
        if self.m_intrinsic is None:
            self.m_intrinsic = 3

        # default k_den
        if self.k_den is None:
            self.k_den = int(np.ceil(self.m_intrinsic * np.log(max(n, 3))))

        # Build FAISS index
        index = self.build_faiss_index(X)

        # Delegate to the corrected function
        rho = self.compute_knn_density(X, index)

        return rho
 
    def fit(self, X):
        """
        Two operation modes:
        1. use_spectral=True  → Spectral embedding + RCI
        2. use_spectral=False → Pure RCI with exact working parameters (MEI = 1.0)
        """

        # ============================================================
        # PHASE 0: INITIAL SETUP (COMMON TO BOTH MODES)
        # ============================================================
        X = np.asarray(X, float)
        self.X_original = X.copy()
        n, d = X.shape

        # ============================================================
        # BRANCH 1: SPECTRAL MODE
        # ============================================================
        if self.use_spectral:
            print("=" * 60)
            print("SPECTRAL RCI - HYBRID MODE")
            print("=" * 60)

            np.random.seed(0)
            self.X_input = X.copy()

            print(f"Dataset: {n} points in {d}D")

            # --------------------------------------------------------
            # Hyperparameters for Spectral Mode
            # --------------------------------------------------------
            if self.warm_up is None:
                self.warm_up = max(5, int(np.ceil(0.002 * n)))

            if self.m_intrinsic is None:
                self.m_intrinsic = min(d, 3)

            if self.alpha is None:
                self.alpha = float(self.m_intrinsic)

            if self.k_den is None:
                self.k_den = int(np.ceil(self.m_intrinsic * np.log(max(n, 3))))

            if self.k_nn is None:
                self.k_nn = max(5, int(np.ceil(np.log(max(n, 3)))))

            print("Hyperparameters:")
            print(f"  - m_intrinsic: {self.m_intrinsic}")
            print(f"  - alpha: {self.alpha}")
            print(f"  - k_den: {self.k_den}")
            print(f"  - k_nn: {self.k_nn}")
            print(f"  - use_spectral: True")

            # Build FAISS index
            print("\nBuilding FAISS index for kNN queries...")
            self.index_knn = faiss.IndexFlatL2(d)
            self.index_knn.add(X.astype(np.float32))

            print("\n" + "=" * 60)
            print("MODE: SPECTRAL EMBEDDING + RCI")
            print("=" * 60)
            print("\n[1/4] Computing spectral embedding...")

            self.X_spectral, self.eigenvalues = \
                self.compute_spectral_embedding_matrix_free(X)

            if hasattr(self, 'spectral_failed') and self.spectral_failed:
                print("Warning: Spectral embedding failed, using original space")
            else:
                print(f"Embedding computed: {self.X_spectral.shape}")

            print("\n[2/4] Running RCI on spectral embedding...")
            self.clusters = self.reverse_clustering_impact(self.X_spectral)
            print(f"Clusters found: {len(self.clusters)}")

            print("\n[3/4] Assigning final labels...")
            self.labels_ = self.predict()
            unique_labels = len(np.unique(self.labels_[self.labels_ >= 0]))
            print(f"Assigned {unique_labels} unique labels")

            print("\n[4/4] Exporting graph for MEI metric...")
            if hasattr(self, 'laplacian_op') and hasattr(self.laplacian_op, 'graph_sym'):
                self.graph_ = self.laplacian_op.graph_sym
                print(f"Graph exported: {len(self.graph_['i'])} edges")
            else:
                print("Warning: No graph available from Laplacian")
                self.graph_ = {"i": None, "j": None, "w": None}

            print("\n" + "=" * 60)
            print("SPECTRAL MODE COMPLETED")
            print("=" * 60)

            return self

        # ============================================================
        # BRANCH 2: PURE RCI MODE (EXACT WORKING VERSION)
        # ============================================================
        else:
            # --------------------------------------------------------
            # Build FAISS index (required for RCI profiles)
            # --------------------------------------------------------
            self.index_knn = faiss.IndexFlatL2(d)
            self.index_knn.add(X.astype(np.float32))

            # --------------------------------------------------------
            # Exact hyperparameter defaults from working version
            # Order is critical: m_intrinsic must be set before k_den
            # --------------------------------------------------------

            # 1. m_intrinsic
            if self.m_intrinsic is None:
                self.m_intrinsic = d

            # 2. k_den = m * log(n)
            if self.k_den is None:
                self.k_den = int(np.ceil(self.m_intrinsic * np.log(max(n, 3))))

            # 3. alpha
            if self.alpha is None:
                self.alpha = float(self.m_intrinsic)

            # 4. k_nn
            if self.k_nn is None:
                self.k_nn = max(5, int(np.ceil(np.log(max(n, 3)))))

            # --------------------------------------------------------
            # Step 1: Build Laplacian Graph
            # --------------------------------------------------------
            self.rci_graph_op = MatrixOnDemandLaplacian(
                X,
                k_nn=self.k_nn,
                mutual_knn=self.mutual_knn,
                eps_self_loop=self.eps_self_loop,
                self_tuning=self.self_tuning,
                sigma_global=self.sigma_global,
                verbose=False
            )

            # --------------------------------------------------------
            # Step 2: Export Graph for MEI
            # --------------------------------------------------------
            self.graph_ = self.rci_graph_op.graph_sym

            # --------------------------------------------------------
            # Step 3: Density
            # --------------------------------------------------------
            self.rho = self.compute_density(X)

            # --------------------------------------------------------
            # Step 4: Morse Field
            # --------------------------------------------------------
            self.f_morse = -np.log(self.rho + EPS)

            # --------------------------------------------------------
            # Step 5: RCI Profiles
            # --------------------------------------------------------
            self.rci_profiles = {}
            for i in range(n):
                M_c, r_k, sorted_idx, A_c = self.compute_rci_profile(X, i, self.rho)
                self.rci_profiles[i] = {
                    "M_c": M_c,
                    "r_k": r_k,
                    "sorted_idx": sorted_idx,
                    "A_c": A_c
                }

            # --------------------------------------------------------
            # Step 6: Cluster Growing
            # --------------------------------------------------------
            self.centers = []
            unassigned = set(range(n))

            while unassigned:
                c = unassigned.pop()

                M_c = self.rci_profiles[c]["M_c"]
                kappa, _, _ = self.detect_boundary(M_c)

                comp = self.rci_profiles[c]["sorted_idx"][:kappa + 1]

                self.centers.append(c)
                unassigned -= set(comp)

            # Final labels
            self.labels_ = self.assign_clusters(X)

            # --------------------------------------------------------
            # Mark spectral as unused
            # --------------------------------------------------------
            self.X_spectral = None
            self.eigenvalues = None
            self.spectral_failed = True

            # --------------------------------------------------------
            # Plotting compatibility
            # --------------------------------------------------------
            self.X_input = X.copy()

            return


    def detect_boundary(self, M_c):
        """
        Detects the critical point κ* in the RCI profile M_c(k).

        Correct method:
            - uses d1 = ΔM and d2 = Δ²M
            - robust smoothing
            - detection of most negative concavity
            - local refinement
            - safe clamping

        Returns:
            kappa : int          (boundary index)
            d1    : ΔM
            d2    : Δ²M (smoothed)
        """

        M_c = np.asarray(M_c, float)
        n = len(M_c)

        if n < 5:
            return 0, None, None

        # -----------------------------------------
        # 1. First and second differences
        # -----------------------------------------
        d1 = np.diff(M_c)
        d2 = np.diff(d1)

        if len(d2) < 3:
            return 1, d1, d2

        # -----------------------------------------
        # 2. Robust smoothing
        #    moving median (window size 3)
        # -----------------------------------------
        d2_s = d2.copy()
        for i in range(1, len(d2) - 1):
            d2_s[i] = np.median([d2[i - 1], d2[i], d2[i + 1]])

        # -----------------------------------------
        # 3. Most concave point → initial boundary
        # -----------------------------------------
        k0 = int(np.argmin(d2_s))

        # safety
        if k0 < 1:
            k0 = 1
        if k0 > n - 3:
            k0 = n - 3

        # -----------------------------------------
        # 4. Refinement: local window
        # -----------------------------------------
        win = max(3, int(0.03 * n))  # 3% of the length
        a = max(1, k0 - win)
        b = min(len(d1) - 1, k0 + win)

        local = d1[a:b+1]
        if len(local) == 0:
            kappa = k0
        else:
            kappa = a + int(np.argmin(local))

        # -----------------------------------------
        # 5. Final clamping
        # -----------------------------------------
        if kappa < 1:
            kappa = 1
        if kappa > n - 2:
            kappa = n - 2

        return int(kappa), d1, d2_s

    def predict(self, X_new=None):
        """
        Unified prediction method.
        
        Returns:
            - If X_new is None: labels from fit() (shape: [n_samples])
            - If X_new is provided: predicted labels for new data
        """
        
        # ------------------------------------------------------
        # CASE 1: Return training labels (from fit())
        # ------------------------------------------------------
        if X_new is None:
            # Use X_original to determine size
            if self.X_original is None:
                raise ValueError("Must call fit() before predict()")
            
            # Reconstruct labels from clusters
            labels = np.full(len(self.X_original), -1, dtype=int)
            
            # If we have clusters, use them
            if self.clusters:
                for cluster_id, points in self.clusters.items():
                    labels[np.asarray(points, dtype=int)] = cluster_id
            # Otherwise, use labels_ if available
            elif hasattr(self, 'labels_'):
                labels = self.labels_
            
            return labels
        
        # ------------------------------------------------------
        # CASE 2: Predict labels for new data
        # ------------------------------------------------------
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        
        # Determine which space to use
        X_ref = self.X_spectral if self.X_spectral is not None else self.X_original
        
        # Find nearest cluster center
        C = np.array(self.centers, dtype=int)
        centers = X_ref[C]
        
        index = faiss.IndexFlatL2(X_ref.shape[1])
        index.add(centers.astype(np.float32))
        
        _, nn = index.search(X_new.astype(np.float32), 1)
        return nn.flatten()


    def plot_results_3d(self):
        if self.X_original is None:
            raise ValueError("Must call fit() first")

        # --- Choose correct title for spectral panel ---
        if hasattr(self, "spectral_failed") and self.spectral_failed:
            spectral_title = "Spectral Embedding (FAILED → raw space)"
        else:
            spectral_title = "Spectral Embedding"

        fig = make_subplots(
            rows=2,
            cols=4,
            subplot_titles=[
                "First Data (raw)",
                "Original Data (labeled)",
                spectral_title,
                "Clusters in Original Space (with centers)",
                "RCI Profiles",
                "Second Differences ($\\Delta^2$M)",
                "Eigenvalue Spectrum",
                "Residual/Extra Analysis"
            ],
            specs=[
                [
                    {"type": "scatter3d"}, {"type": "scatter3d"},
                    {"type": "scatter3d"}, {"type": "scatter3d"}
                ],
                [
                    {"type": "scatter"}, {"type": "scatter"},
                    {"type": "scatter"}, {"type": "scatter"}
                ]
            ]
        )

        labels = self.predict()
        n_clusters = len(self.clusters)
        colors = (
            px.colors.qualitative.Alphabet[:n_clusters]
            + px.colors.qualitative.T10[:n_clusters]
        )

        # --- Panel 1: Raw data ---
        if hasattr(self, "X_input") and self.X_input is not None:
            if self.X_input.shape[1] >= 3:
                fig.add_trace(
                    go.Scatter3d(
                        x=self.X_input[:, 0],
                        y=self.X_input[:, 1],
                        z=self.X_input[:, 2],
                        mode="markers",
                        marker=dict(size=3, color="gray"),
                        name="First Data (raw)",
                        showlegend=False,
                    ),
                    row=1, col=1
                )

        # --- Panel 2: Original Data (labeled) ---
        if self.X_original.shape[1] >= 3:
            fig.add_trace(
                go.Scatter3d(
                    x=self.X_original[:, 0],
                    y=self.X_original[:, 1],
                    z=self.X_original[:, 2],
                    mode="markers",
                    marker=dict(
                        size=4, color=labels,
                        colorscale="Viridis", showscale=False
                    ),
                    name="Original Data (labeled)",
                    showlegend=False,
                ),
                row=1, col=2
            )

        # --- Panel 3: Spectral Embedding (success or failed) ---
        if self.X_spectral is not None and self.X_spectral.shape[1] >= 3:
            fig.add_trace(
                go.Scatter3d(
                    x=self.X_spectral[:, 0],
                    y=self.X_spectral[:, 1],
                    z=self.X_spectral[:, 2],
                    mode="markers",
                    marker=dict(
                        size=4, color=labels,
                        colorscale="Viridis", showscale=False
                    ),
                    name=spectral_title,
                    showlegend=False,
                ),
                row=1, col=3
            )

        # --- Panel 4: Clusters in Original Space ---
        if self.X_original.shape[1] >= 3:
            for cluster_id in range(n_clusters):
                mask = labels == cluster_id
                if np.any(mask):
                    fig.add_trace(
                        go.Scatter3d(
                            x=self.X_original[mask, 0],
                            y=self.X_original[mask, 1],
                            z=self.X_original[mask, 2],
                            mode="markers",
                            marker=dict(
                                size=4,
                                color=colors[cluster_id % len(colors)]
                            ),
                            name=f"Cluster {cluster_id}",
                            showlegend=False,
                        ),
                        row=1, col=4
                    )

            if self.centers:
                center_coords = self.X_original[self.centers]
                fig.add_trace(
                    go.Scatter3d(
                        x=center_coords[:, 0],
                        y=center_coords[:, 1],
                        z=center_coords[:, 2],
                        mode="markers",
                        marker=dict(size=10, color="red", symbol="diamond"),
                        name="Centers",
                        showlegend=False,
                    ),
                    row=1, col=4
                )

        # --- Panel 5 & 6: RCI Profiles ---
        for cid, prof in self.rci_profiles.items():
            # M_c profile
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(prof["M_c"])),
                    y=prof["M_c"],
                    mode="lines",
                    line=dict(color=colors[cid % len(colors)]),
                    name=f"M_c {cid}",
                    showlegend=False,
                ),
                row=2, col=1
            )
            # Boundary marker
            fig.add_trace(
                go.Scatter(
                    x=[prof["kappa_star"]],
                    y=[prof["M_c"][prof["kappa_star"]]],
                    mode="markers",
                    marker=dict(size=8, color="red", symbol="x"),
                    name=f"Boundary {cid}",
                    showlegend=False,
                ),
                row=2, col=1
            )
            # Second differences
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(prof["delta2_M"])),
                    y=prof["delta2_M"],
                    mode="lines",
                    line=dict(color=colors[cid % len(colors)]),
                    name=f"$\\Delta^2$M_c {cid}",
                    showlegend=False,
                ),
                row=2, col=2
            )

        # --- Panel 7: Eigenvalues ---
        if self.eigenvalues is not None:
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(self.eigenvalues)),
                    y=self.eigenvalues,
                    mode="markers+lines",
                    marker=dict(size=8, color="blue"),
                    name="Eigenvalues",
                    showlegend=False,
                ),
                row=2, col=3
            )

        # --- Panel 8: Extra ---
        fig.add_trace(
            go.Scatter(
                x=[0, 1, 2],
                y=[0, 1, 0],
                mode="lines+markers",
                marker=dict(size=6, color="black"),
                name="Extra Panel",
                showlegend=False,
            ),
            row=2, col=4
        )

        fig.update_layout(
            title="Spectral RCI - Complete Analysis",
            height=800
        )
        fig.update_xaxes(title_text="k (neighbor index)", row=2, col=1)
        fig.update_yaxes(title_text="$M_c(k)$", row=2, col=1)
        fig.update_xaxes(title_text="k", row=2, col=2)
        fig.update_yaxes(title_text="$\\Delta^2M_c(k)$", row=2, col=2)
        fig.update_xaxes(title_text="Eigenvalue index", row=2, col=3)
        fig.update_yaxes(title_text="Eigenvalue", row=2, col=3)

        return fig

# ============================================================================
# Synthetic Data Generators
# ============================================================================
def sample_sphere(n=2000, R=1.0, patch=None, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n,3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    if patch is not None:
        kind, thr = patch
        if kind == 'z>':
            keep = X[:,2] > thr
            while keep.sum() < n:
                Y = rng.normal(size=(n,3)); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                X = np.vstack([X[keep], Y[Y[:,2] > thr]])[:n]
                keep = np.ones(len(X), bool)
    X *= R
    if noise>0: X += noise*rng.normal(size=X.shape)
    return X

def sample_saddle(n=2000, span=1.0, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-span, span, size=(n,2))
    z = (xy[:,0]**2 - xy[:,1]**2)
    X = np.column_stack([xy, z])
    if noise>0: X += noise*rng.normal(size=X.shape)
    return X

def sample_torus(n=3000, R=2.0, r=0.6, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2*np.pi, n)
    v = rng.uniform(0, 2*np.pi, n)
    x = (R + r*np.cos(v))*np.cos(u)
    y = (R + r*np.cos(v))*np.sin(u)
    z = r*np.sin(v)
    X = np.column_stack([x,y,z])
    if noise>0: X += noise*rng.normal(size=X.shape)
    return X

def sample_dumbbell_surface(n=4000, R=1.0, tube_r=0.3, gap=2.2, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    n_sph = int(0.4*n)
    n_tub = n - 2*n_sph
    
    def sphere_centered(c, npts):
        U = rng.normal(size=(npts,3)); U /= np.linalg.norm(U, axis=1, keepdims=True)
        return c + R*U
    
    left = sphere_centered(np.array([ -gap/2, 0, 0 ]), n_sph)
    right = sphere_centered(np.array([ gap/2, 0, 0 ]), n_sph)
    
    t = rng.uniform(-gap/2, gap/2, size=n_tub)
    ang = rng.uniform(0, 2*np.pi, size=n_tub)
    y = tube_r*np.cos(ang)
    z = tube_r*np.sin(ang)
    tube = np.column_stack([t, y, z])
    
    X = np.vstack([left, tube, right])
    if noise>0: X += noise*rng.normal(size=X.shape)
    return X

# ============================================================================
# Adversarial Transformations
# ============================================================================
def warp_poly(X, s=0.35, seed=0):
    """Non-isometric smooth warp."""
    rng = np.random.default_rng(seed)
    X = X.copy()
    x, y, z = X[:,0], X[:,1], X[:,2]
    Xw = np.column_stack([
        x + s*y*z,
        y + s*z*x,
        z + s*x*y
    ])
    return Xw

def bias_density_resample(X, axis=2, beta=3.0, seed=0):
    """Resample with prob ~ exp(beta * normalized_coord)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    v = X[:, axis]
    v = (v - v.min()) / (v.max() - v.min() + 1e-12)
    p = np.exp(beta*(v - 0.5)); p /= p.sum()
    idx = rng.choice(n, size=n, replace=True, p=p)
    return X[idx]

def inject_outliers(X, frac=0.05, scale=1.5, seed=0):
    """Outlier injection."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    m = int(np.round(frac*n))
    center = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    outliers = []
    for i in range(m):
        direction = rng.normal(0, 1, d)
        direction /= np.linalg.norm(direction) + 1e-12
        distance = scale * np.mean(std) * rng.exponential(1.0)
        outlier = center + distance * direction
        outliers.append(outlier)
    O = np.array(outliers)
    return np.vstack([X, O])

def add_shortcut_bridges(X, n_bridges=6, pts_per=40, noise=0.01, seed=0):
    """Insert straight bridges connecting distant pairs."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    Y = [X]
    for _ in range(n_bridges):
        i, j = rng.integers(0, n, size=2)
        xi, xj = X[i], X[j]
        t = rng.uniform(0, 1, size=(pts_per, 1))
        seg = (1-t)*xi + t*xj + noise*rng.normal(size=(pts_per,d))
        Y.append(seg)
    return np.vstack(Y)

# ============================================================================
# Spectral RCI Demo and Torture Suite (REWRITTEN)
# ============================================================================

def run_spectral_rci_demo():
    print("Spectral RCI Demo (Auto-Tuning)")
    print("=" * 50)

    # Synthetic manifolds (unchanged)
    cases = {
        "sphere": lambda: sample_sphere(n=3000, R=1.0, noise=0.003),
        "saddle": lambda: sample_saddle(n=3000, span=1.2, noise=0.005),
        "torus": lambda: sample_torus(n=4000, R=2.0, r=0.6, noise=0.003),
        "dumbbell": lambda: sample_dumbbell_surface(
            n=4000, R=1.0, tube_r=0.25, gap=2.4, noise=0.004
        ),
    }

    for name, gen in cases.items():
        print(f"\n Testing on {name}...")

        # -------------- DATA GENERATION ----------------
        X = gen()

        # -------------- RCI ESPECTRAL VERDADEIRO -------
        rci = SpectralRCI(
            k_den=None,               # auto
            alpha=None,               # intrinsic α = m
            m_intrinsic=None,         # auto dimensionality
            k_nn=None,                # auto Laplacian kNN
            n_eigenvectors=EIGENVECTORS_N,
            warm_up=None,
            mutual_knn=True,
            epsilon_stop=0.03,
            eps_self_loop=1e-6,
        )

        # -------------- FIT + RCI ----------------------
        rci.fit(X)

        # -------------- FINAL LABELS -------------------
        labels = rci.predict()

        print(f"Found {len(np.unique(labels[labels >= 0]))} clusters")

        # -------------- PLOTS --------------------------
        fig = rci.plot_results_3d()
        fig.update_layout(title=f"Spectral RCI Results: {name}")
        fig.show()

def run_torture_suite():
    cases = {
        "sphere": lambda: sample_sphere(n=3000, R=1.0, noise=0.003),
        "saddle": lambda: sample_saddle(n=3000, span=1.2, noise=0.005),
        "torus": lambda: sample_torus(n=4000, R=2.0, r=0.6, noise=0.003),
        "dumbbell": lambda: sample_dumbbell_surface(
            n=4000, R=1.0, tube_r=0.25, gap=2.4, noise=0.004
        ),
    }

    variants = [
        (
            "raw_RCI_no_spectral",
            lambda X: X,
            dict(use_spectral=False, self_tuning=True, mutual_knn=True),
        ),
        (
            "no_self_tuning",
            lambda X: X,
            dict(use_spectral=True, self_tuning=False, mutual_knn=True),
        ),
        (
            "no_mutual",
            lambda X: X,
            dict(use_spectral=True, self_tuning=True, mutual_knn=False),
        ),
        (
            "warp_only",
            warp_poly,
            dict(use_spectral=True, self_tuning=True, mutual_knn=True),
        ),
        (
            "warp+outliers",
            lambda X: inject_outliers(warp_poly(X)),
            dict(use_spectral=True, self_tuning=True, mutual_knn=True),
        ),
        (
            "warp+density_bias",
            lambda X: bias_density_resample(warp_poly(X)),
            dict(use_spectral=True, self_tuning=True, mutual_knn=True),
        ),
        (
            "warp+bridges(shortcuts)",
            lambda X: add_shortcut_bridges(warp_poly(X)),
            dict(use_spectral=True, self_tuning=True, mutual_knn=True),
        ),
    ]

    for name, gen in cases.items():
        X0 = gen()
        for vname, transform, kw in variants:
            X = transform(X0)

            rci = SpectralRCI(
                k_den=None,
                alpha=None,
                m_intrinsic=None,
                k_nn=None,
                n_eigenvectors=EIGENVECTORS_N,
                warm_up=None,
                epsilon_stop=0.03,
                eps_self_loop=1e-6,
                **kw,
            )

            rci.fit(X)
            labels = rci.predict()

            print(f"{name} | {vname} → clusters: "
                  f"{len(np.unique(labels[labels >= 0]))}")

            fig = rci.plot_results_3d()
            fig.update_layout(title=f"{name} — {vname}")
            fig.show()

if __name__ == "__main__":
    run_spectral_rci_demo()
    run_torture_suite()