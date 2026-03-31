# RCI: Reverse Clustering Impact

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-research-red.svg)](https://github.com/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17759236.svg)](https://doi.org/10.5281/zenodo.17759236)

> **Official implementation of "Reverse Clustering Impact: Geometry-Driven Minimal Parameters for Clustering via Morse Theory"**

> KMeans slices a torus into wedges. DBSCAN fragments a trefoil.
> RCI reads the geometry — with **one parameter**.

<p align="center">
  <img src="images/hero_comparison.png" width="900" alt="KMeans and DBSCAN fail on non-linear geometry; RCI detects intrinsic structure">
</p>

---

## 🚀 Run in 30 seconds

```bash
git clone https://github.com/Regis3336/rci-clustering.git
cd rci-clustering
pip install -r requirements.txt
python -m rci.core
```

→ Produces clustering + geometry diagnostics immediately

---

## 🧪 Minimal example

```python
from rci.core import SpectralRCI
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=500, noise=0.05)

model = SpectralRCI(r=0.1)
model.fit(X)
labels = model.predict()

print(labels[:10])
```

---

## 📖 What is RCI?

RCI is a geometric clustering method that detects intrinsic structure in non-linear data.

Unlike classical methods:

* **KMeans** imposes Euclidean geometry
* **DBSCAN** requires careful density tuning

RCI instead reads the **intrinsic curvature** of the data manifold.

The algorithm requires **only one parameter** $r$.
All structural quantities — intrinsic dimension, density thresholds, curvature bounds — are inferred automatically from local geometry.

---

## 🧠 Core ideas

**Curvature Law** — cluster boundaries are detected via the curvature-sensitive signature $\Delta^2 M_c(k)$, which is negative on positively-curved regions and positive on saddles.

**Matrix-on-Demand Spectral Engine** — custom `MatrixOnDemandLaplacian` performs spectral embedding without constructing dense $N \times N$ matrices, scaling gracefully to large datasets.

**Morse Erasure Index (MEI)** — a parameter-free intrinsic metric that quantifies how much of the Morse density field persists across cluster boundaries. Used as the primary evaluation metric throughout.

---

## 📊 Benchmark results

RCI achieves the highest structural fidelity in **7 of 8 datasets**.

<table align="center">
  <thead>
    <tr>
      <th>Dataset</th><th>RCI</th><th>KMeans</th><th>DBSCAN</th><th>HDBSCAN</th><th>Spectral</th><th>GMM</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Sphere</td>     <td><b>0.70</b></td><td>0.12</td><td>0.56</td><td>0.44</td><td>0.10</td><td>0.13</td></tr>
    <tr><td>Saddle</td>     <td>0.03</td><td>0.06</td><td><b>0.52</b></td><td>0.41</td><td>0.07</td><td>0.07</td></tr>
    <tr><td>Torus</td>      <td><b>0.74</b></td><td>0.03</td><td>0.00</td><td>0.44</td><td>0.03</td><td>0.03</td></tr>
    <tr><td>Dumbbell</td>   <td><b>0.75</b></td><td>0.00</td><td>0.26</td><td>0.27</td><td>0.00</td><td>0.00</td></tr>
    <tr><td>Link</td>       <td><b>0.47</b></td><td>0.04</td><td>0.02</td><td>0.10</td><td>0.03</td><td>0.03</td></tr>
    <tr><td>Spiral</td>     <td><b>0.61</b></td><td>0.03</td><td>0.03</td><td>0.30</td><td>0.03</td><td>0.03</td></tr>
    <tr><td>Swiss Roll</td> <td><b>0.61</b></td><td>0.02</td><td>0.17</td><td>0.04</td><td>0.04</td><td>0.02</td></tr>
    <tr><td>Trefoil</td>    <td><b>0.68</b></td><td>0.03</td><td>0.05</td><td>0.37</td><td>0.03</td><td>0.03</td></tr>
  </tbody>
  <tfoot>
    <tr><td><b>Mean</b></td>   <td><b>0.57</b></td><td>0.04</td><td>0.20</td><td>0.30</td><td>0.04</td><td>0.04</td></tr>
    <tr><td><b>Std Dev</b></td><td>0.22</td><td>0.03</td><td>0.21</td><td>0.14</td><td>0.03</td><td>0.04</td></tr>
  </tfoot>
</table>

<p align="center">
  <em>MEI-based structural comparison. RCI achieves the highest mean (0.57), winning 7 of 8 datasets.</em>
</p>

Run the full benchmark suite:

```bash
python -m benchmarks.comparison_suite
```

---

## 🔬 Reproducibility & validation

### Spectral RCI pipeline

<p align="center">
  <img src="images/sphere_validation.png" width="850" alt="Spectral RCI on the sphere dataset">
</p>

Top: raw point cloud, RCI labels, spectral embedding, farthest-point centers.
Bottom: Morse profiles, curvature signature, eigenvalue spectrum, diagnostics.

---

### Structural homology validation (Appendix C)

`homology.py` verifies the theory computationally:

* reconstructs the Čech nerve
* checks the Scale Sheaf Axioms
* matches persistence with merge transitions

```bash
python -m rci.homology
```

<p align="center">
  <img src="images/cluster_evolution.png" width="800" alt="Topological Cluster Evolution">
</p>

---

## 📚 Theoretical foundation

📄 **[theoretical_foundation/RCI_foundation.pdf](theoretical_foundation/RCI_foundation.pdf)**

Includes:

* Curvature Law
* Multi-scale Morse framework
* Farthest-point geometry
* MEI construction

---

## 🧩 Implementation and theory

The code is a **computational instantiation** of the mathematical framework, not a heuristic approximation.

The symbolic theory and the implementation were developed independently — their agreement is verified internally via `homology.py`.

Related framework:

👉 https://github.com/Regis3336/operational-coherence

---

## 🧭 Roadmap

* **Graph-RCI** — extension to weighted graphs using spectral Laplacians
* **Hypergraph-RCI** — extension to higher-order incidence structures
* **Temporal-RCI (RCI-T)** — dynamic clustering with time-indexed Morse profiles

Full log: [CHANGELOG.md](CHANGELOG.md)

---

## 📂 Repository structure

```text
rci-clustering/
├── rci/
│   ├── core.py                  # Main algorithm (SpectralRCI, MatrixOnDemandLaplacian)
│   └── homology.py              # Theoretical validation (Sheaf, Nerve, Persistence)
├── theoretical_foundation/
│   └── RCI_foundation.pdf       # Full mathematical framework
├── benchmarks/
│   └── comparison_suite.py      # Benchmark suite
├── generate_hero_image.py       # Generates images/hero_comparison.png
├── images/                      # Figures and documentation assets
├── results/                     # Scoreboard from the benchmark
└── CHANGELOG.md                 # Development log
```

---

## 📄 Citation

```bibtex
@misc{junior2025rci,
  title        = {Reverse Clustering Impact: Geometry-Driven Minimal Parameters for Clustering via Morse Theory},
  author       = {Junior, Reinaldo Elias de Souza},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17759236},
  url          = {https://doi.org/10.5281/zenodo.17759236}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE).
