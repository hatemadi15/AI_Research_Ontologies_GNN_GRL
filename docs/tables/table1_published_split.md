Table 1. Published splits, 10 model seeds: mean F1 [95% CI]; differences paired by seed. The published score is plain MatSciBERT's on MaterioMiner (Durmaz et al. 2024) and MatSciBERT-CRF's on SOFC-Exp (Gupta et al. 2022).

| split | system | micro | macro | weighted | strict micro | published |
|---|---|---|---|---|---|---|
| MaterioMiner, fine-grained | plain MatSciBERT | 70.19 [69.03, 71.35] | 67.73 [65.85, 69.60] | 70.53 [69.34, 71.73] | 71.61 [70.51, 72.72] | 69.92 |
| MaterioMiner, fine-grained | CRF head | 73.31 [72.28, 74.35] | 71.84 [70.14, 73.54] | 72.75 [71.65, 73.86] | 73.31 [72.28, 74.35] |  |
| MaterioMiner, fine-grained | CRF head − plain | +3.12 [+1.62, +4.62] | +4.11 [+1.77, +6.45] | +2.22 [+0.66, +3.78] | +1.70 [+0.21, +3.19] |  |
| MaterioMiner, coarse-grained | plain MatSciBERT | 72.52 [71.88, 73.16] | 60.66 [59.09, 62.24] | 71.83 [71.18, 72.47] | 74.40 [73.54, 75.27] | 72.32 |
| MaterioMiner, coarse-grained | CRF head | 75.48 [74.23, 76.72] | 64.33 [61.97, 66.68] | 74.64 [73.30, 75.99] | 75.48 [74.23, 76.72] |  |
| MaterioMiner, coarse-grained | CRF head − plain | +2.96 [+1.30, +4.61] | +3.66 [+0.90, +6.43] | +2.82 [+1.13, +4.50] | +1.07 [−0.82, +2.97] |  |
| SOFC-Exp | plain MatSciBERT | 82.53 [81.84, 83.23] | 80.15 [79.44, 80.86] | 82.60 [81.91, 83.29] | 83.02 [82.38, 83.65] |  |
| SOFC-Exp | CRF head | 83.75 [82.85, 84.65] | 81.11 [80.02, 82.20] | 83.72 [82.77, 84.67] | 83.75 [82.85, 84.65] | 84.77 (macro 82.39) |
| SOFC-Exp | CRF head − plain | +1.22 [+0.50, +1.93] | +0.96 [+0.16, +1.76] | +1.12 [+0.37, +1.86] | +0.73 [−0.04, +1.51] |  |
| SOFC-Exp | `crf_ctx` − CRF head | +0.58 [−0.26, +1.42] | +0.85 [−0.20, +1.90] | +0.63 [−0.22, +1.49] | +0.58 [−0.26, +1.42] |  |
| SOFC-Exp | `crf_dc` − CRF head | −0.42 [−0.61, −0.22] | −0.48 [−0.70, −0.25] | −0.31 [−0.52, −0.10] | −0.42 [−0.61, −0.22] |  |
| SOFC-Exp | `crf_dcr` − CRF head | −0.38 [−0.60, −0.15] | −0.41 [−0.68, −0.14] | −0.27 [−0.50, −0.04] | −0.38 [−0.60, −0.15] |  |
| SOFC-Exp | `crf_ebv` − CRF head | −0.21 [−0.58, +0.17] | −0.14 [−0.47, +0.20] | −0.17 [−0.51, +0.17] | −0.21 [−0.58, +0.17] |  |
