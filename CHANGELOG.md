# Changelog

All notable changes to **RCI: Reverse Clustering Impact** are documented here.  
This project follows a structural-behavioral versioning logic rather than semantic versioning.

---

## [Unreleased]

Planned extensions and research directions:

- **Graph-RCI:** extension of the farthest-point cover, Morse profiles, and curvature signature to general weighted graphs.
- **Hypergraph-RCI:** generalization to multi-way relations and higher-order incidence structures.
- **Temporal-RCI:** dynamic Morse profiles, curvature flow over time, and MEI stability for evolving datasets.

---

## [1.1.0] — Mathematical Foundations Update — 2025-11-29

### Updated

- **Section 4 of the theoretical foundation fully revised:**  
  Integrated full proofs for:
  - small-ball regularity and population \(k\)-NN scaling;  
  - uniform \(C^0\)-conjugacy between radius and mass potentials;  
  - discrete Morse stability lemma;  
  - geometric bounds for doubling spaces;  
  - nerve-theoretic topological transitions;  
  - comparison theorem \(M(r)\) vs. \(N(r)\);  
  - full proof of fractal completeness of \(M(r)\);  
  - heuristic local–global scale correspondence (Proposition 4.19) and growth-exponent detection (Corollary 4.20).

- **Paper quality improvements:**  
  Removed placeholder text (“Proof identical to…”), resolved formatting issues, ensured all referenced results are self-contained in the PDF.

### Added

- Expanded and clarified connections between curvature signatures, doubling geometry, population radii, and fractal completeness.
- Updated `theorical_foundation/RCI_foundation.pdf` with complete mathematical arguments.

---

## [1.0.0] — Initial Release — 2025-11-27

### Added

- **SpectralRCI core (`rci/core.py`):**
  - matrix-on-demand Laplacian
  - farthest-point geometric cover
  - intrinsic dimension detection
  - curvature signature \(\Delta^2 M_c(k)\)
  - auto-tuned spectral embedding
  - generation of `images/sphere_validation.png`

- **Homology validation module (`rci/homology.py`):**
  - Čech nerve construction
  - scale-sheaf consistency checks
  - \(H_0\) persistence barcode reconstruction
  - generation of `images/cluster_evolution.png`

- **Benchmark suite (`benchmarks/comparsion_suite.py`):**
  - MEI computation
  - evaluation across 8 geometric datasets
  - export of `results/scoreboard.csv`

- **Documentation:**
  - full README with theory, diagrams, commands, and MEI scoreboard
  - reproducibility instructions for module-based execution
  - integration of figures produced by `core.py` and `homology.py`

- **Repository layout:**
  - `rci/`, `benchmarks/`, `images/`, `results/`, `theorical_foundation/` directories
  - versioned mathematical foundations in `theorical_foundation/RCI_foundation.pdf`

---

## [0.1.0] — Pre-release Internal Iterations

Private exploratory versions containing early Morse signatures, prototype Laplacian, and preliminary MEI experiments.
