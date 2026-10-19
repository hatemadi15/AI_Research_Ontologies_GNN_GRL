Table 3. Default (conlleval) against strict IOB2 scoring, published splits, 10 model seeds: micro F1 [95% CI].

| split | system | default | strict IOB2 |
|---|---|---|---|
| MaterioMiner, fine-grained | plain MatSciBERT | 70.19 [69.03, 71.35] | 71.61 [70.51, 72.72] |
| MaterioMiner, fine-grained | CRF head | 73.31 [72.28, 74.35] | 73.31 [72.28, 74.35] |
| MaterioMiner, fine-grained | CRF head − plain | +3.12 [+1.62, +4.62] | +1.70 [+0.21, +3.19] |
| MaterioMiner, coarse-grained | plain MatSciBERT | 72.52 [71.88, 73.16] | 74.40 [73.54, 75.27] |
| MaterioMiner, coarse-grained | CRF head | 75.48 [74.23, 76.72] | 75.48 [74.23, 76.72] |
| MaterioMiner, coarse-grained | CRF head − plain | +2.96 [+1.30, +4.61] | +1.07 [−0.82, +2.97] |
| SOFC-Exp | plain MatSciBERT | 82.53 [81.84, 83.23] | 83.02 [82.38, 83.65] |
| SOFC-Exp | CRF head | 83.75 [82.85, 84.65] | 83.75 [82.85, 84.65] |
| SOFC-Exp | CRF head − plain | +1.22 [+0.50, +1.93] | +0.73 [−0.04, +1.51] |
| SOFC-Exp, slot types | plain MatSciBERT | 70.20 [68.82, 71.58] | 71.94 [70.69, 73.19] |
| SOFC-Exp, slot types | CRF head | 71.88 [71.01, 72.75] | 71.88 [71.01, 72.75] |
| SOFC-Exp, slot types | CRF head − plain | +1.68 [−0.12, +3.49] | −0.06 [−1.70, +1.58] |
