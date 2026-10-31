Table 9. Post hoc robustness of the pooled comparisons (ner_robustness.py): over the 11 held-out units of decision 13.2 (MuLMS's 7 subfields and MaterioMiner's 4 fine-grained papers), every comparison among the CRF head, mention replacement with 3 copies (MR3), with 6 copies (MR6) and MR6 with harvested names, with the MuLMS units as 10-seed means (as decided) and as means over the 7 seeds the screen did not see; the mean paired difference with its 95% Student-t CI, the number of units above 0, the exact two-sided sign-flip permutation p-value of the mean, and for the names against the CRF head the lowest lower bound over the 11 leave-one-unit-out intervals. Only decision 13.2 (first row of the first basis) was pre-registered.

| MuLMS seed basis | comparison | Δ micro F1 [95% CI], 11 units | units above 0 | sign-flip p | leave-one-unit-out: lowest lower bound |
|---|---|---|---|---|---|
| MuLMS seeds 0–900 (as decided) | MR6 + names − CRF head | +3.58 [+0.95, +6.21] | 9/11 | 0.0107 | +0.44 |
| MuLMS seeds 0–900 (as decided) | MR6 + names − MR3 | +1.23 [−0.22, +2.68] | 8/11 | 0.0898 |  |
| MuLMS seeds 0–900 (as decided) | MR6 + names − MR6 | +1.02 [−0.56, +2.59] | 6/11 | 0.1836 |  |
| MuLMS seeds 0–900 (as decided) | MR6 − CRF head | +2.56 [+0.94, +4.18] | 10/11 | 0.0049 |  |
| MuLMS seeds 0–900 (as decided) | MR6 − MR3 | +0.22 [−0.30, +0.74] | 8/11 | 0.3672 |  |
| MuLMS seeds 0–900 (as decided) | MR3 − CRF head | +2.35 [+0.94, +3.75] | 10/11 | 0.0029 |  |
| MuLMS seeds 300–900 (not seen by the screen) | MR6 + names − CRF head | +3.58 [+1.04, +6.13] | 10/11 | 0.0068 | +0.54 |
| MuLMS seeds 300–900 (not seen by the screen) | MR6 + names − MR3 | +1.13 [−0.51, +2.76] | 7/11 | 0.1582 |  |
| MuLMS seeds 300–900 (not seen by the screen) | MR6 + names − MR6 | +0.80 [−0.66, +2.26] | 7/11 | 0.2383 |  |
| MuLMS seeds 300–900 (not seen by the screen) | MR6 − CRF head | +2.79 [+1.20, +4.38] | 10/11 | 0.0020 |  |
| MuLMS seeds 300–900 (not seen by the screen) | MR6 − MR3 | +0.33 [−0.25, +0.92] | 8/11 | 0.2334 |  |
| MuLMS seeds 300–900 (not seen by the screen) | MR3 − CRF head | +2.46 [+1.00, +3.91] | 11/11 | 0.0010 |  |


**MaterioMiner's held-out papers (10 seeds)**

| comparison | fine-grained, 4 papers [95% CI] | coarse-grained, 4 papers [95% CI] |
|---|---|---|
| MR6 + names − CRF head | +4.27 [+1.39, +7.15] | +1.77 [+0.67, +2.88] |
| MR6 + names − MR3 | +1.50 [−1.76, +4.75] | +0.49 [+0.06, +0.92] |
| MR6 + names − MR6 | +1.29 [−2.51, +5.10] | +0.70 [−1.60, +3.00] |
| MR6 − CRF head | +2.98 [+0.09, +5.86] | +1.07 [−1.82, +3.97] |
| MR6 − MR3 | +0.20 [−1.47, +1.88] | −0.21 [−2.71, +2.29] |
| MR3 − CRF head | +2.77 [+1.51, +4.03] | +1.28 [+0.06, +2.51] |
