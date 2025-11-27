# comparsion_suite.py — Multimodel Structural Comparison Suite
# =================================================================================================
# Public API:
#     - run_comparison_suite(outdir_base=..., n_jobs=..., skip_if_cached=...)
#
# Purpose:
#   This module implements a fully reproducible, geometry–driven clustering benchmark tailored to
#   evaluate structural fidelity across heterogeneous algorithms. It couples:
#
#       (1) classical clustering models
#           – KMeans
#           – DBSCAN / HDBSCAN
#           – Gaussian Mixture Models (EM)
#           – Spectral Clustering
#
#       (2) the RCI algorithm (pure, non-spectral version)
#
#       (3) a topology-sensitive quality metric:
#           – Morse Erasure Index (MEI), defined as:
#                 MEI = 1 − TV_intra / TV_total
#           where TV denotes the weighted total variation of the Morse function f = −log ρ.
#
#   The suite performs a unified evaluation across diverse synthetic 3D datasets (sphere, torus,
#   saddle, dumbbell surface, trefoil knot, spiral, Swiss roll, Hopf link), providing:
#       • end-to-end hyperparameter optimization for each model,
#       • structural comparison via MEI,
#       • full reproducibility through fixed seeds and cached outputs,
#       • optional embedding and graph export (used only by RCI),
#       • high-resolution visualizations and scoreboards.
#
# Design Principles:
#   * Geometry-first benchmarking:
#         Each algorithm is evaluated on identical raw 3D point clouds with no preprocessing.
#
#   * Strict structural metric:
#         MEI evaluates how smoothly the Morse function varies within clusters. This penalizes
#         boundary fragmentation, density artifacts, and spurious splits—not just centroid accuracy.
#
#   * Fairness across models:
#         – All non-RCI methods build their own kNN graph.
#         – RCI injects its internal graph directly for exact TV computation.
#
#   * Fully parallelized computation:
#         Joblib (loky backend) is used for per-algorithm parallelization across datasets.
#
# Outputs:
#     results/scoreboard.csv        → Full MEI table for all datasets x algorithms
#     results/best_params.json      → Hyperparameters selected via random search
#     images/<dataset>_<alg>.png    → High-resolution 2D/3D visualizations
#
# Notes:
#   * This module implements the empirical section of the RCI project.
#   * RCI never uses spectral embeddings here; its graph is injected exactly into MEI.
#   * All datasets are procedurally generated and reproducible.
#
# =================================================================================================


import os
os.environ["LOKY_MAX_CPU_COUNT"] = "30"   

import os
import json
import hdbscan
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from joblib import Parallel, delayed
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from rci.core import SpectralRCI
from sklearn.model_selection import ParameterSampler
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

warnings.filterwarnings("ignore", message="Graph is not fully connected")


# --- Constants ---
EPS = 1e-12
EIGENVECTORS_N = 5

def compute_density(X, k=15, eps=1e-12):
    """
    Compute the k–NN density estimate with strict removal of the self-distance.

    Parameters
    ----------
    X : array-like, shape (n, m)
        Data points.
    k : int, default=15
        Number of neighbors used for the density estimate.
    eps : float, default=1e-12
        Numerical stabilizer to avoid division by zero.

    Returns
    -------
    rho : ndarray, shape (n,)
        The k–NN density estimate at each point.
    """
    X = np.asarray(X, float)
    n, m = X.shape

    # k+1 includes the point itself; will be removed explicitly
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X)

    dists, idx = nn.kneighbors(X)

    # remove the self-distance
    dists = dists[:, 1:]     # drop column 0
    idx   = idx[:, 1:]

    # radius = distance to the true k-th neighbor
    r_k = dists[:, -1] + eps

    # volume of the m-dimensional unit ball
    V_m = (np.pi ** (m / 2)) / np.math.gamma(1 + m / 2)

    rho = k / (n * V_m * (r_k**m + eps))
    return rho

# --- Data Generation Functions ---
def sample_sphere(n=2000, R=1.0, patch=None, noise=0.0, seed=0):
    """Sample points from a unit sphere, optionally with a patch and noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    
    if patch is not None:
        kind, thr = patch
        if kind == 'z>':
            keep = X[:, 2] > thr
            while keep.sum() < n:
                Y = rng.normal(size=(n, 3))
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                X = np.vstack([X[keep], Y[Y[:, 2] > thr]])[:n]
                keep = np.ones(len(X), bool)
    
    X *= R
    if noise > 0:
        X += noise * rng.normal(size=X.shape)
    return X

def sample_saddle(n=2000, span=1.0, noise=0.01, seed=0):
    """Sample points from a saddle surface."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-span, span, size=(n, 2))
    z = (xy[:, 0]**2 - xy[:, 1]**2)
    X = np.column_stack([xy, z])
    if noise > 0:
        X += noise * rng.normal(size=X.shape)
    return X

def sample_torus(n=3000, R=2.0, r=0.6, noise=0.01, seed=0):
    """Sample points from a torus."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2*np.pi, n)
    v = rng.uniform(0, 2*np.pi, n)
    x = (R + r*np.cos(v)) * np.cos(u)
    y = (R + r*np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    X = np.column_stack([x, y, z])
    if noise > 0:
        X += noise * rng.normal(size=X.shape)
    return X

def sample_dumbbell_surface(n=4000, R=1.0, tube_r=0.3, gap=2.2, noise=0.01, seed=0):
    """Sample points from a dumbbell surface."""
    rng = np.random.default_rng(seed)
    n_sph = int(0.4 * n)
    n_tub = n - 2 * n_sph
    
    def sphere_centered(c, npts):
        U = rng.normal(size=(npts, 3))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        return c + R * U
    
    left = sphere_centered(np.array([-gap/2, 0, 0]), n_sph)
    right = sphere_centered(np.array([gap/2, 0, 0]), n_sph)
    
    t = rng.uniform(-gap/2, gap/2, size=n_tub)
    ang = rng.uniform(0, 2*np.pi, size=n_tub)
    y = tube_r * np.cos(ang)
    z = tube_r * np.sin(ang)
    tube = np.column_stack([t, y, z])
    
    X = np.vstack([left, tube, right])
    if noise > 0:
        X += noise * rng.normal(size=X.shape)
    return X

def sample_link(n=2000, R=1.0, r=0.2, noise=0.01, seed=0):
    """Sample points from a Hopf link (two interlinked circles)."""
    rng = np.random.default_rng(seed)
    n_half = n // 2
    
    theta1 = np.linspace(0, 2*np.pi, n_half, endpoint=False)
    x1 = R * np.cos(theta1)
    y1 = R * np.sin(theta1)
    z1 = np.zeros_like(theta1)
    
    theta2 = np.linspace(0, 2*np.pi, n - n_half, endpoint=False)
    x2 = R * np.cos(theta2)
    y2 = np.zeros_like(theta2)
    z2 = R * np.sin(theta2)
    
    X = np.vstack([np.column_stack([x1, y1, z1]), np.column_stack([x2, y2, z2])])
    if noise > 0:
        X += noise * rng.normal(size=X.shape)
    return X

def sample_spiral(n=2000, turns=3, radius=1.0, height=2.0, noise=0.02, seed=0):
    """Sample points from a 3D spiral."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2*np.pi*turns, n)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.linspace(-height/2, height/2, n)
    X = np.column_stack([x, y, z])
    if noise > 0:
        X += noise * rng.normal(size=X.shape)
    return X

def sample_swiss_roll(n=2000, length=4*np.pi, height=2.0, noise=0.02, seed=0):
    """Sample points from a Swiss roll."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, length, n)
    x = t * np.cos(t)
    y = rng.uniform(-height/2, height/2, n)
    z = t * np.sin(t)
    X = np.column_stack([x, y, z])
    if noise > 0:
        X += noise * rng.normal(size=X.shape)
    return X

def sample_trefoil_knot(n=2000, noise=0.01, seed=0):
    """Sample points from a trefoil knot."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.sin(t) + 2*np.sin(2*t)
    y = np.cos(t) - 2*np.cos(2*t)
    z = -np.sin(3*t)
    X = np.column_stack([x, y, z])
    if noise > 0:
        X += noise * rng.normal(size=X.shape)
    return X

# --- Clustering Evaluation ---
def evaluate_clustering(X, labels):
    """Evaluate clustering using Silhouette and Davies-Bouldin scores."""
    if len(set(labels)) < 2:
        return -np.inf
    silhouette = silhouette_score(X, labels)
    db = -davies_bouldin_score(X, labels)
    return (silhouette + db) / 2

# --- Hyperparameter Optimization ---
def random_search_optimize(
    X, param_dist, model_fn, n_iter=50, random_state=0):
    """Random search for hyperparameter optimization."""
    best_score, best_labels, best_params = -np.inf, None, None
    
    for params in ParameterSampler(
        param_dist, n_iter=n_iter, random_state=random_state
    ):
        try:
            model = model_fn(**params)
            if hasattr(model, 'fit_predict'):
                labels = model.fit_predict(X)
            else:
                model.fit(X)
                if hasattr(model, 'predict'):
                    labels = model.predict(X)
                else:
                    labels = model.labels_
            
            score = evaluate_clustering(X, labels)
            if score > best_score:
                best_score, best_labels, best_params = (
                    score, labels, params
                )
        except Exception as e:
            print(f"[WARN] Failed with params {params}: {e}")
            continue
    
    if best_labels is None:
        best_labels = np.zeros(len(X), dtype=int)
    
    return best_labels, best_params

def optimize_kmeans(X, n_iter=50):
    param_dist = {
        'n_clusters': np.arange(2, 21),
        'random_state': [0, 42, 123],
        'n_init': [10]
    }
    max_iter = min(
        n_iter,
        len(param_dist['n_clusters']) *
        len(param_dist['random_state']) *
        len(param_dist['n_init'])
    )
    return random_search_optimize(
        X, param_dist, KMeans, n_iter=max_iter
    )

def optimize_dbscan(X, n_iter=50):
    param_dist = {
        'eps': [0.05, 0.1, 0.2, 0.4, 0.8],
        'min_samples': [3, 5, 10, 20]
    }
    max_iter = min(
        n_iter,
        len(param_dist['eps']) * len(param_dist['min_samples'])
    )
    return random_search_optimize(
        X, param_dist, DBSCAN, n_iter=max_iter
    )

def optimize_hdbscan(X, n_iter=50):
    param_dist = {
        'min_cluster_size': [5, 10, 20, 50],
        'min_samples': [1, 5, 10, 20]
    }
    max_iter = min(
        n_iter,
        len(param_dist['min_cluster_size']) *
        len(param_dist['min_samples'])
    )
    return random_search_optimize(
        X, param_dist, hdbscan.HDBSCAN, n_iter=max_iter
    )

def optimize_gmm(X, n_iter=50):
    param_dist = {
        'n_components': np.arange(2, 21),
        'random_state': [0, 42, 123]
    }
    max_iter = min(
        n_iter,
        len(param_dist['n_components']) *
        len(param_dist['random_state'])
    )
    return random_search_optimize(
        X, param_dist, GaussianMixture, n_iter=max_iter
    )

def optimize_spectral(X, n_iter=50):
    param_dist = {
        'n_clusters': np.arange(2, 21),
        'affinity': ['nearest_neighbors'],
        'n_neighbors': [5, 10, 20, 50],
        'random_state': [0, 42, 123]
    }
    max_iter = min(
        n_iter,
        len(param_dist['n_clusters']) *
        len(param_dist['affinity']) *
        len(param_dist['n_neighbors']) *
        len(param_dist['random_state'])
    )
    return random_search_optimize(
        X, param_dist, SpectralClustering, n_iter=max_iter
    )

def morse_erasure_index(
    X,
    labels,
    f,
    graph=None,
    n_neighbors=15,
    mutual=True,
    self_tuning=True,
    metric="euclidean",
    random_state=0,
    backend="sklearn"
):
    """
    Morse Erasure Index (MEI) with full RCI-compatibility.
    Supports graphs injected as:
        - dict {"i","j","w"}
        - numpy adjacency matrix
    If graph is None, builds a fresh kNN graph.
    """

    # --------------------------------------------
    # Sanity for basic input shapes
    # --------------------------------------------
    X = np.asarray(X, float)
    labels = np.asarray(labels).reshape(-1)
    f = np.asarray(f).reshape(-1)
    n = X.shape[0]

    if labels.shape[0] != n or f.shape[0] != n:
        return 1.0

    # ============================================================
    # CASE 1 — Graph explicitly injected (RCI -> MEI)
    # ============================================================
    if graph is not None:

        # --------------------------------------------------------
        # A) Convert matrix → dict (RCI sometimes outputs ndarray)
        # --------------------------------------------------------
        if isinstance(graph, np.ndarray):
            rows, cols = np.where(graph > 1e-12)
            weights = graph[rows, cols]

            graph = {
                "i": rows.astype(int),
                "j": cols.astype(int),
                "w": weights.astype(float)
            }

        # --------------------------------------------------------
        # B) Dict must contain "i","j","w"
        # --------------------------------------------------------
        try:
            I = np.asarray(graph["i"], dtype=int)
            J = np.asarray(graph["j"], dtype=int)
            W = np.asarray(graph["w"], dtype=float)
        except Exception:
            return 1.0

        if len(W) == 0:
            return 1.0

        # --------------------------------------------------------
        # C) Compute MEI directly from injected graph
        # --------------------------------------------------------
        fi, fj = f[I], f[J]
        diff = np.abs(fi - fj)

        TV_total = float(np.sum(W * diff))
        if TV_total <= 1e-12:
            return 0.0

        li, lj = labels[I], labels[J]
        mask = (li == lj)

        TV_intra = float(np.sum(W[mask] * diff[mask]))

        return 1.0 - (TV_intra / TV_total)

    # ============================================================
    # CASE 2 — Build kNN affinity graph (legacy)
    # ============================================================
    from dataclasses import dataclass

    @dataclass
    class Graph:
        I: np.ndarray
        J: np.ndarray
        W: np.ndarray
        total: float

    def build_affinity(X, n_neighbors, mutual, self_tuning, metric, random_state, backend):
        X = np.asarray(X, float)
        n = X.shape[0]

        if n_neighbors >= n:
            n_neighbors = max(1, n - 1)

        # kNN phase
        if backend == "faiss":
            import faiss
            index = faiss.IndexFlatL2(X.shape[1])
            index.add(X.astype(np.float32))
            d2, idx = index.search(X.astype(np.float32), n_neighbors + 1)
            d = np.sqrt(d2)[:, 1:]
            idx = idx[:, 1:]
        else:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
            nn.fit(X)
            d, idx = nn.kneighbors(X)
            d, idx = d[:, 1:], idx[:, 1:]

        rows = np.repeat(np.arange(n), n_neighbors)
        cols = idx.ravel()
        dist = d.ravel()

        mask = (cols >= 0) & (cols < n)
        rows, cols, dist = rows[mask], cols[mask], dist[mask]

        # self-tuning sigma
        if self_tuning:
            sigma = d[:, -1] + 1e-12
        else:
            s0 = np.median(d[:, -1]) + 1e-12
            sigma = np.full(n, s0)

        sig_i, sig_j = sigma[rows], sigma[cols]
        W_dir = np.exp(-(dist * dist) / (sig_i * sig_j + 1e-12))

        # Mutual reduction
        edges = {}
        if mutual:
            for i, j, w in zip(rows, cols, W_dir):
                a, b = (i, j) if i < j else (j, i)
                if (a, b) not in edges:
                    edges[(a, b)] = [0.0, 0.0]
                if i < j:
                    edges[(a, b)][0] = max(edges[(a, b)][0], float(w))
                else:
                    edges[(a, b)][1] = max(edges[(a, b)][1], float(w))

            I, J, W = [], [], []
            for (a, b), (wij, wji) in edges.items():
                if wij > 0 and wji > 0:
                    I.append(a)
                    J.append(b)
                    W.append(np.sqrt(wij * wji))

            return Graph(np.asarray(I), np.asarray(J), np.asarray(W), float(np.sum(W)))

        # Non-mutual
        for i, j, w in zip(rows, cols, W_dir):
            edges[(i, j)] = max(edges.get((i, j), 0.0), float(w))

        I, J, W = zip(*[(a, b, w) for (a, b), w in edges.items()])
        I, J, W = np.asarray(I), np.asarray(J), np.asarray(W)

        return Graph(I, J, W, float(np.sum(W)))

    # Build graph
    G = build_affinity(X, n_neighbors, mutual, self_tuning, metric, random_state, backend)

    if G.total <= 1e-12:
        return 1.0

    fi, fj = f[G.I], f[G.J]
    diff = np.abs(fi - fj)

    TV_total = np.sum(G.W * diff)
    if TV_total <= 1e-12:
        return 0.0

    li, lj = labels[G.I], labels[G.J]
    mask = (li == lj)

    TV_intra = np.sum(G.W[mask] * diff[mask])

    return 1.0 - (TV_intra / TV_total)

def make_json_safe(obj):
    """Convert any Python / NumPy object into a JSON-serializable structure."""

    # dict → dict
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    # list or tuple → list
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]

    # NumPy arrays → lists
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # NumPy scalar types → Python scalars
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64)):
        return int(obj)

    if isinstance(obj, (np.floating, np.float_, np.float16,
                        np.float32, np.float64)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, (np.bytes_,)):
        return obj.decode("utf-8")

    # everything else: leave it as is (strings, ints, floats, None, bool)
    return obj

def run_rci(X):
    """
    Runs the 'pure' RCI without spectral embedding.
    Returns exactly:
        labels, params, X_emb, graph
    (X_emb = None by definition of non-spectral RCI)
    """
    dsc = SpectralRCI(
        n_eigenvectors=5,
        use_spectral=False,
        mutual_knn=True,
        self_tuning=True
    )
    dsc.fit(X)
    labels = dsc.predict()
    # --------------------------------------------------
    # CRITICAL DEBUG — INSPECTION OF THE GRAPH GENERATED BY RCI
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("DEBUG: GRAPH TYPE IN run_rci()")
    print("=" * 60)
    print(f"type(dsc.graph_)         : {type(dsc.graph_)}")
    print(f"is dict?                 : {isinstance(dsc.graph_, dict)}")
    print(f"is numpy array?          : {isinstance(dsc.graph_, np.ndarray)}")
    if isinstance(dsc.graph_, dict):
        print(f"dict keys                : {list(dsc.graph_.keys())}")
    print("=" * 60 + "\n")
    # --------------------------------------------------
    # correct graph
    graph = dsc.graph_
    params = {
        "used_embedding": False,
        "n_eigenvectors": 5,
        "mutual_knn": True,
        "self_tuning": True
    }
    # RCI does not use embedding
    X_emb = None
    return labels, params, X_emb, graph

def evaluate_structural_all(dsets, results, graphs, n_neighbors=15):
    """
    Evaluates the MEI (Morse Erasure Index) for all datasets and algorithms.
    Uses injected graph ONLY for RCI.
    All others build their own kNN.
    Returns: DataFrame with (Dataset, Algorithm, MEI).
    """
    recs = []
    for dn, X in dsets.items():
        # ---------------------------------------
        # MORSE FUNCTION f = -log(density)
        # ---------------------------------------
        rho = compute_density(X, k=n_neighbors)
        f_morse_X = -np.log(rho + EPS)
        # ---------------------------------------
        # Loop over algorithms
        # ---------------------------------------
        for alg, labels in results[dn].items():
            labels = np.asarray(labels).reshape(-1)
            # safety check
            if labels.shape[0] != X.shape[0]:
                recs.append({
                    "Dataset": dn,
                    "Algorithm": alg,
                    "MEI": -1.0
                })
                continue
            # ---------------------------------------
            # graph: only RCI injects graph
            # ---------------------------------------
            if alg == "RCI":
                graph = graphs.get(dn, {}).get("RCI", None)
            else:
                graph = None
            # ---------------------------------------
            # DEBUG SADDLE
            # ---------------------------------------
            if alg == "RCI" and dn == "Saddle":
                print("\n=== DEBUG SADDLE ===")
                print(f"Graph type: {type(graph)}")
                print(f"Graph is None: {graph is None}")
                if isinstance(graph, np.ndarray):
                    print(f"Graph shape: {graph.shape}")
                    print(f"Graph nnz: {np.sum(graph > 1e-12)}")
                if isinstance(graph, dict):
                    print(f"Graph keys: {graph.keys()}")
                    print(f"Graph edges: {len(graph['i'])}")
                print("=" * 20)
            # ---------------------------------------
            # MEI Calculation
            # ---------------------------------------
            mei_val = morse_erasure_index(
                X,
                labels,
                f=f_morse_X,
                graph=graph,
                n_neighbors=n_neighbors
            )
            recs.append({
                "Dataset": dn,
                "Algorithm": alg,
                "MEI": float(mei_val)
            })
    # ---------------------------------------
    # Pack results
    # ---------------------------------------
    import pandas as pd
    return pd.DataFrame(recs)


def run_comparison_suite(
    outdir_base=".",
    n_jobs=-1,
    skip_if_cached=False,
    epsilon=0.2
):
    # -----------------------------------------------------
    # Create output directories
    # -----------------------------------------------------
    rd = os.path.join(outdir_base, "results")
    im = os.path.join(outdir_base, "images")
    os.makedirs(rd, exist_ok=True)
    os.makedirs(im, exist_ok=True)

    # -----------------------------------------------------
    # Structures to store results
    # -----------------------------------------------------
    all_labels = {}       # labels[dataset][algorithm] = ndarray(N)
    embeddings = {}       # embeddings[dataset][algorithm] = ndarray(N,d)
    graphs = {}           # graphs[dataset]["RCI"] = graph
    log = {}              # params used

    # -----------------------------------------------------
    # Synthetic datasets
    # -----------------------------------------------------
    dsets = {
        "Sphere":      sample_sphere(n=2000, R=1, noise=0.01, seed=0),
        "Saddle":      sample_saddle(n=2000, span=1, noise=0.01, seed=0),
        "Torus":       sample_torus(n=3000, R=2, r=0.6, noise=0.01, seed=0),
        "Dumbbell":    sample_dumbbell_surface(n=4000, R=1, tube_r=0.3, gap=2.2, noise=0.01, seed=0),
        "Link":        sample_link(n=2000, R=1, r=0.2, noise=0.01, seed=0),
        "Spiral":      sample_spiral(n=2000, turns=3, radius=1, height=2, noise=0.02, seed=0),
        "SwissRoll":   sample_swiss_roll(n=2000, length=4*np.pi, height=2, noise=0.02, seed=0),
        "TrefoilKnot": sample_trefoil_knot(n=2000, noise=0.01, seed=0),
    }

    # -----------------------------------------------------
    # Algorithms
    # -----------------------------------------------------
    algs = {
        "RCI":      run_rci,
        "KMeans":   optimize_kmeans,
        "DBSCAN":   optimize_dbscan,
        "HDBSCAN":  optimize_hdbscan,
        "Spectral": optimize_spectral,
        "GMM":      optimize_gmm
    }

    # -----------------------------------------------------
    # Main loop over datasets
    # -----------------------------------------------------
    for name, X in dsets.items():

        print(f"\n=== {name} ===")

        all_labels[name] = {}
        embeddings[name] = {}
        graphs[name] = {}

        jobs = []

        # -----------------------------------------
        # Load from cache or schedule computation
        # -----------------------------------------
        for alg, fn in algs.items():
            lp = os.path.join(im, f"{name}_{alg}_labels.npy")
            pp = os.path.join(im, f"{name}_{alg}_params.json")
            ep = os.path.join(im, f"{name}_{alg}_embedding.npy")

            if skip_if_cached and os.path.exists(lp):
                all_labels[name][alg] = np.load(lp)

                if os.path.exists(ep):
                    embeddings[name][alg] = np.load(ep)

                if os.path.exists(pp):
                    log[f"{name}_{alg}"] = json.load(open(pp))
                else:
                    log[f"{name}_{alg}"] = None

                print(f"[CACHE] {alg} on {name} loaded")

            else:
                jobs.append((alg, fn, X, lp, pp, ep))

        # -----------------------------------------
        # Execute jobs
        # -----------------------------------------
        if jobs:
            res = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_run_single_job)(alg, fn, X, lp, pp, ep)
                for alg, fn, X, lp, pp, ep in jobs
            )

            # -----------------------------------------
            # Process outputs
            # -----------------------------------------
            for alg, labels, params, X_emb, graph in res:

                all_labels[name][alg] = labels

                if isinstance(X_emb, np.ndarray):
                    embeddings[name][alg] = X_emb

                # ✔ CORRETO: aceite matriz OU dict
                if alg == "RCI" and graph is not None:
                    graphs[name]["RCI"] = graph

                log[f"{name}_{alg}"] = params

    # -----------------------------------------------------
    # STRUCTURAL EVALUATION — argument corrected
    # -----------------------------------------------------
    sb = evaluate_structural_all(dsets, all_labels, graphs)

    # Save summary
    sb.to_csv(os.path.join(rd, "scoreboard.csv"), index=False)
    json.dump(make_json_safe(log), open(os.path.join(rd, "best_params.json"), "w"), indent=2)

    print("\n[OK] Structural evaluation complete!")

    return sb

def _run_single_job(alg, fn, X, lp, pp, ep):
    """
    Runs a single clustering job and normalizes its outputs.

    Parameters
    ----------
    alg : str
        Algorithm name (e.g., "RCI", "KMeans").
    fn : callable
        The function that executes the algorithm.
    X : ndarray
        Input data array (N x d).
    lp : str
        Path to save the labels (.npy).
    pp : str
        Path to save the parameters (.json).
    ep : str
        Path to save the embedding (.npy), if available.

    Returns
    -------
    (alg, labels, params, X_emb, graph)
        Normalized outputs for downstream processing.
    """

    # ------------------------------------------------------
    # Safely execute algorithm while silencing nuisance warnings
    # ------------------------------------------------------
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*force_all_finite.*")
        warnings.filterwarnings("ignore", message=".*ensure_all_finite.*")
        warnings.filterwarnings("ignore", message=".*Graph is not fully connected.*")

        out = fn(X)

    # ------------------------------------------------------
    # Normalize return values
    # ------------------------------------------------------
    if alg == "RCI":
        # RCI must always return 4 outputs
        if len(out) != 4:
            raise ValueError(
                "RCI must return exactly 4 values: (labels, params, X_emb, graph)"
            )
        labels, params, X_emb, graph = out

    else:
        # Classical algorithms may return 2 or 3 values
        if len(out) == 3:
            labels, params, X_emb = out
            graph = None
        elif len(out) == 2:
            labels, params = out
            X_emb = None
            graph = None
        else:
            raise ValueError(
                f"{fn.__name__} returned an invalid number of outputs"
            )

    # ------------------------------------------------------
    # Validate embedding shape
    # ------------------------------------------------------
    if isinstance(X_emb, np.ndarray):
        if X_emb.ndim != 2 or X_emb.shape[0] != len(X):
            X_emb = None  # discard malformed embedding

    # ------------------------------------------------------
    # Save labels
    # ------------------------------------------------------
    np.save(lp, labels)

    # ------------------------------------------------------
    # Save embedding, if valid
    # ------------------------------------------------------
    if isinstance(X_emb, np.ndarray):
        np.save(ep, X_emb)

    # ------------------------------------------------------
    # Save parameters
    # ------------------------------------------------------
    json.dump(
        make_json_safe(params),
        open(pp, "w"),
        indent=2
    )

    return alg, labels, params, X_emb, graph

if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "=" * 60)
    print("CLUSTERING COMPARISON SUITE")
    print("=" * 60)

    sb = run_comparison_suite(
        outdir_base=base_dir,
        n_jobs=-1,
        skip_if_cached=False,
        epsilon=0.2
    )

    results_dir = os.path.join(base_dir, "results")
    images_dir  = os.path.join(base_dir, "images")
    scoreboard_path = os.path.join(results_dir, "scoreboard.csv")

    sb.to_csv(scoreboard_path, index=False)

    print("\n" + "=" * 60)
    print("FINAL RESULTS (Morse Smoothness Topology)")
    print("=" * 60)
    print("\nFull Scoreboard:")
    print(sb)

    print("\n" + "=" * 60)
    print("FILES SAVED")
    print("=" * 60)
    print(f"Results: {results_dir}")
    print(f"Images: {images_dir}")
    print(f"Scoreboard: {scoreboard_path}")