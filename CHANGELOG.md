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

## [1.0.0] — Initial Release — 2025-11-27

### Added

- **SpectralRCI core (`rci/core.py`):**
    
    - matrix-on-demand Laplacian
        
    - farthest-point geometric cover
        
    - intrinsic dimension detection
        
    - curvature signature Δ2Mc(k)\Delta^{2} M_c(k)Δ2Mc​(k)
        
    - auto-tuned spectral embedding
        
    - generation of `images/sphere_validation.png`
        
- **Homology validation module (`rci/homology.py`):**
    
    - Čech nerve construction
        
    - scale-sheaf consistency checks
        
    - H0H_0H0​ persistence barcode reconstruction
        
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
    
    - `rci/`, `benchmarks/`, `images/`, `results/`, `theory/` directories
        
    - versioned mathematical foundations in `theory/RCI_foundation.pdf`
        

---

## [0.1.0] — Pre-release Internal Iterations

Private exploratory versions containing early Morse signatures, prototype Laplacian, and preliminary MEI experiments.
