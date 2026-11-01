Table 16. The t statistic and P values of paper A's decisions (D5' the post hoc reading of D5 on MuLMS's decision seeds): a one-sample Student-t test of the mean paired difference against 0 over the units, n - 1 degrees of freedom, with the standard deviation (SD) of the per-unit differences, from the per-unit differences the decision used (rounded to two decimals, so t and P can differ from the unrounded ones in the last digit), the exact two-sided sign-flip P over all 2^n sign patterns beside it. Reporting only: each decision was taken by the registered rule, the 95% interval above 0 (for the pooled ones, with neither corpus's interval below 0), which is the one-sided test at 2.5%.

| decision | comparison | units | Δ [95% CI] | SD | t (df) | P one-sided | P two-sided | sign-flip P | verdict |
|---|---|---|---|---|---|---|---|---|---|
| D1 | MuLMS: MR6 + names vs CRF head | 7 | +3.19 [−1.13, +7.51] | 4.67 | 1.81 (6) | 0.0603 | 0.1207 | 0.0781 | no |
| D2 | MuLMS: MR6 + names vs MR3 | 7 | +0.92 [−1.63, +3.48] | 2.76 | 0.88 (6) | 0.2062 | 0.4125 | 0.4062 | no |
| D3 | MaterioMiner fine: MR6 + names vs CRF head | 4 | +4.27 [+1.39, +7.15] | 1.81 | 4.72 (3) | 0.0090 | 0.0180 | 0.1250 | passes |
| D4 | MaterioMiner coarse: MR6 + names vs CRF head | 4 | +1.77 [+0.67, +2.88] | 0.69 | 5.12 (3) | 0.0072 | 0.0144 | 0.1250 | passes |
| D5 | pooled 11 units (MuLMS seeds 0-900): MR6 + names vs CRF head | 11 | +3.58 [+0.95, +6.21] | 3.92 | 3.03 (10) | 0.0063 | 0.0127 | 0.0107 | passes |
| D5' | pooled 11 units (MuLMS seeds 300-900, post hoc reading): MR6 + names vs CRF head | 11 | +3.58 [+1.04, +6.13] | 3.79 | 3.14 (10) | 0.0053 | 0.0106 | 0.0068 | passes |
| D6 | SOFC-Exp folds: MR6 + names vs CRF head | 5 | −0.04 [−1.21, +1.14] | 0.94 | −0.08 (4) | 0.5284 | 0.9432 | 1.0000 | no |
| D7 | pooled 11 units: MR6 + names vs MR6 | 11 | +0.80 [−0.66, +2.26] | 2.17 | 1.22 (10) | 0.1258 | 0.2517 | 0.2383 | no |
| D8 | MuLMS: own-topic vs mixed corpus | 7 | −0.18 [−2.01, +1.65] | 1.98 | −0.24 (6) | 0.5926 | 0.8148 | 0.7969 | no |
| D9 | MaterioMiner fine: multi-task + MR6 + names vs multi-task + MR6 | 4 | +0.96 [−0.01, +1.92] | 0.61 | 3.14 (3) | 0.0259 | 0.0517 | 0.1250 | no |
| D10 | MaterioMiner coarse: multi-task + MR6 + names vs multi-task + MR6 | 4 | +0.85 [−0.72, +2.42] | 0.99 | 1.72 (3) | 0.0916 | 0.1833 | 0.2500 | no |
| D11 | MuLMS: MR6 + names vs recall-matched MR6, recall on the unseen names the lexicon holds | 7 | +11.24 [+3.31, +19.18] | 8.58 | 3.47 (6) | 0.0067 | 0.0133 | 0.0156 | passes |
| D12 | MuLMS: MR6 + names vs recall-matched MR6, micro F1 | 7 | +3.26 [−0.79, +7.31] | 4.38 | 1.97 (6) | 0.0480 | 0.0961 | 0.1406 | no |
