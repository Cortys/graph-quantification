# Implementation for Adjusted Count Quantification Learning on Graphs

The implementation of quantification is split across multiple modules:
- [`data/quantification.py`](./gq/data/quantification.py): Implementation for synthetic distribution shift (PPS, BFS- and PPR-based covariate shift).
- [`nn/quantification_metrics.py`](./gq/nn/quantification_metrics.py): Contains implementations for generic quantification methods (CC, ACC, DMy, KDEy). The implementation makes use of the QuaPy library.
- [`nn/graph_quantification_metrics.py`](./gq/nn/graph_quantification_metrics.py): Contains implementations of NACC and SIS for ACC (PPR and SP kernels).
