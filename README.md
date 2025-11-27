# RCI: Reverse Clustering Impact

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-research-red.svg)](https://github.com/)

> **Official implementation of "Reverse Clustering Impact: Geometry-Driven Minimal Parameters for Clustering via Morse Theory"**

<p align="center">
  <img 
    src="images/sphere_validation_v2.png" 
    width="850"
    alt="Spectral RCI analysis on a sphere: embedding, labels, Morse profiles and curvature signatures"
  >
</p>

---

## 📖 Overview

**RCI** is a parameter-minimal geometric clustering framework. Unlike classical methods (KMeans, DBSCAN) that rely on heuristic hyperparameters, RCI couples local geometric variation with global multi-scale structure using **Discrete Morse Theory**.

The algorithm requires only a single operational scale parameter, $r$. All remaining structural quantities—such as intrinsic dimension, density thresholds, and curvature bounds—are determined automatically by small-ball asymptotics and nearest-neighbor statistics.

### Key Contributions
*   **Curvature-Driven:** Detects cluster boundaries via a curvature-sensitive signature $\Delta^2 M_c(k)$, which is negative on positive curvature regions and positive on saddles.
*   **Matrix-on-Demand Spectral Engine:** Includes a custom `MatrixOnDemandLaplacian` that performs spectral embedding without ever constructing dense $N \times N$ matrices, scaling to large datasets.
*   **Morse Erasure Index (MEI):** Introduces a parameter-free intrinsic metric to evaluate structural fidelity by quantifying how much of the Morse density field persists across cluster boundaries.

---

## 📚 Theoretical Foundation

The full mathematical foundation of RCI is provided in the included paper:

📄 **[theoretical_foundation/RCI_foundation.pdf](theoretical_foundation/RCI_foundation.pdf)**

This document presents:
1.  The derivation of the **Curvature Law**: $\Delta^2 M_c(k) \approx - C_d \, S(x_c) \, r_k^2$.
2.  The multi-scale **Morse framework**.
3.  The discrete **farthest-point geometry** underlying the global coverage.
4.  The formal construction of the **MEI metric**.

It serves as the official reference for all theoretical claims made in this repository.

---

## 🧩 Operational Coherence: Why RCI Is Not a Simulation

The implementation in this repository is not a numerical illustration or a heuristic simulation. It is an independent computational instantiation whose structural identity with the symbolic theory has been formally verified using the **[Operational Coherence Framework](https://github.com/Regis3336/operational-coherence)**.

Under this framework, we verify:

1.  **Independence:** The symbolic RCI theory (Syn) and the Python implementation (Comp) were constructed in disjoint categories, preventing circular validity.
2.  **Yoneda Operational Triangle:** The implementation and theory produce identical actions across a wide range of probes (datasets, perturbations, scales), implying structural identity.
3.  **Kolmogorov Rigidity:** The probability of accidental agreement is $< 10^{-300}$, rendering convergence a mathematical certainty rather than an empirical observation.
4.  **Cohomological Verification:** The obstruction class $[T - I] \in H^0(X, \mathcal{E})$ vanishes, meaning there is no topological barrier preventing the code from being the exact object described by the theory.

**Conclusion:** Every experiment, figure, and metric in this repository is a **computable instantiation** of the RCI theory. The curvature signature $\Delta^2 M_c(k)$ in the code is the same object as in the theorem.

*(To reproduce this verification, see the `operational-coherence` repository).*

---

## 🛠 Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Regis3336/rci-clustering.git
cd rci-clustering
pip install -r requirements.txt
```

**Requirements:** `numpy`, `scipy`, `scikit-learn`, `faiss-cpu` (or gpu), `plotly`, `pandas`, `hdbscan`, `networkx`.

---

## ⚡ Quick Start

RCI is designed to be a drop-in replacement for standard clustering workflows.

```python
from rci.core import SpectralRCI, sample_torus

# 1. Generate Synthetic Data (e.g., a Torus)
X = sample_torus(n=3000, R=2.0, r=0.6, noise=0.01)

# 2. Initialize RCI
# Note: RCI auto-tunes intrinsic dimension and density kernels.
model = SpectralRCI(
    n_eigenvectors=5,
    use_spectral=True,
    mutual_knn=True
)

# 3. Fit and Predict
model.fit(X)
labels = model.predict()

# 4. Visualize Results (opens interactive Plotly dashboard)
fig = model.plot_results_3d()
fig.show()
```

---

## 🔬 Reproducibility & Validation

This repository provides the full suite required to reproduce all experiments and mathematical validations presented in the paper.

### 1. Structural Homology Validation (Appendix C)

We offer a computational verification of the sheaf-theoretic foundations of RCI. The script below checks the **Scale Sheaf Axioms**, constructs the **Čech Nerve** of the spectral cover, and confirms that the $H_0$ persistence barcode matches the algorithmic merge profile.

<p align="center">
  <img 
    src="images/cluster_evolution_v2.png" 
    width="800" 
    alt="Topological Cluster Evolution"
  >
</p>

**Run the theory validation suite:**
```bash
python -m rci.homology
```
*Output: Verifies Separatedness, Gluing, Functoriality, and Nerve Theorem compliance.*

### 2. Benchmark Comparison
Compare RCI against classical algorithms (KMeans, DBSCAN, HDBSCAN, Spectral Clustering, GMM) using the **MEI** metric across 8 geometric datasets.

```bash
python -m benchmarks.comparison_suite
```
*Output: Generates `results/scoreboard.csv`, reproduces figure images in `images/`, and computes MEI scores.*

<table align="center">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>RCI</th>
      <th>KMeans</th>
      <th>DBSCAN</th>
      <th>HDBSCAN</th>
      <th>Spectral</th>
      <th>GMM</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Sphere</td>     <td><b>0.70</b></td> <td>0.12</td> <td>0.56</td> <td>0.44</td> <td>0.10</td> <td>0.13</td></tr>
    <tr><td>Saddle</td>     <td>0.03</td> <td>0.06</td> <td><b>0.52</b></td> <td>0.41</td> <td>0.07</td> <td>0.07</td></tr>
    <tr><td>Torus</td>      <td><b>0.74</b></td> <td>0.03</td> <td>0.00</td> <td>0.44</td> <td>0.03</td> <td>0.03</td></tr>
    <tr><td>Dumbbell</td>   <td><b>0.75</b></td> <td>0.00</td> <td>0.26</td> <td>0.27</td> <td>0.00</td> <td>0.00</td></tr>
    <tr><td>Link</td>       <td><b>0.47</b></td> <td>0.04</td> <td>0.02</td> <td>0.10</td> <td>0.03</td> <td>0.03</td></tr>
    <tr><td>Spiral</td>     <td><b>0.61</b></td> <td>0.03</td> <td>0.03</td> <td>0.30</td> <td>0.03</td> <td>0.03</td></tr>
    <tr><td>Swiss Roll</td> <td><b>0.61</b></td> <td>0.02</td> <td>0.17</td> <td>0.04</td> <td>0.04</td> <td>0.02</td></tr>
    <tr><td>Trefoil</td>    <td><b>0.68</b></td> <td>0.03</td> <td>0.05</td> <td>0.37</td> <td>0.03</td> <td>0.03</td></tr>
  </tbody>
  <tfoot>
    <tr><td><b>Mean</b></td>    <td><b>0.57</b></td> <td>0.04</td> <td>0.20</td> <td>0.30</td> <td>0.04</td> <td>0.04</td></tr>
    <tr><td><b>Std Dev</b></td> <td>0.22</td> <td>0.03</td> <td>0.21</td> <td>0.14</td> <td>0.03</td> <td>0.04</td></tr>
  </tfoot>
</table>

<p align="center">
  <em>
    Table 1 — MEI-based structural comparison across benchmark datasets.
    RCI achieves the highest mean MEI (0.57), winning in 7 out of 8 datasets.
  </em>
</p>

---

## 🧭 Changelog & Roadmap

The complete evolution log of the project is available in: **[CHANGELOG.md](CHANGELOG.md)**.

### Roadmap — Upcoming Extensions
The next development phase focuses on extending RCI from point-cloud geometry to combinatorial and temporal structures:

*   **Graph-RCI:** Generalization of Morse curvature and MEI to weighted graphs, leveraging spectral graph Laplacians.
*   **Hypergraph-RCI:** Extension of farthest-point geometry to higher-order incidence structures, with MEI defined over simplicial weight flows.
*   **Temporal-RCI (RCI-T):** A dynamic version for evolving datasets using time-indexed Morse profiles to detect structural transitions (drifts).

---

## 📂 Repository Structure

```text
rci-clustering/
├── rci/
│   ├── core.py              # Main algorithm (SpectralRCI, MatrixFreeLaplacian)
│   └── homology.py          # Theoretical validation (Sheaf, Nerve, Persistence)
├── theoretical_foundation/
│   └── RCI_foundation.pdf   # Full mathematical framework
├── benchmarks/
│   └── comparison_suite.py  # Benchmarking vs. sklearn/hdbscan
├── images/                  # Figures and documentation assets
├── results/                 # Scoreboards, logs, theory validation output
└── CHANGELOG.md             # Development log and ongoing evolution
```

## 📄 Citation

If you use RCI or the MEI metric in your research, please cite:

```bibtex
@article{souza2025rci,
  title={Reverse Clustering Impact: Geometry-Driven Minimal Parameters for Clustering via Morse Theory},
  author={Reinaldo Elias de Souza Junior},
  journal={arXiv preprint},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
