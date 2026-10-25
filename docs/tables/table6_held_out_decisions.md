Table 6. Pre-registered decisions on held-out documents of three corpora from weeks 7 to 13: MuLMS's held-out subfields, the consistency rule, the week-12 winner's MaterioMiner analogue, SOFC-Exp's article folds and the pooled decision (GPU runs, 10 model seeds; a unit is a held-out subfield, paper or article fold, each side the mean over its seeds; MaterioMiner's other unseen-paper decisions are in Table 2): mean paired difference of micro F1 over the units with its 95% CI, and whether the CI lies above 0.

| week | corpus, protocol | comparison | decision | units | Δ micro F1 [95% CI] | CI above 0 |
|---|---|---|---|---|---|---|
| 7 | MaterioMiner fine-grained, unseen papers | `crf_dc` vs `crf` | week 7 (recomputed) | 4 | +0.30 [−0.35, +0.95] | no |
| 7 | MaterioMiner coarse-grained, unseen papers | `crf_dc` vs `crf` | week 7 (recomputed) | 4 | +0.12 [+0.01, +0.24] | passes |
| 10 | MuLMS, held-out subfields | `crf_mr` vs `crf` | week 10, decision 1 | 7 | +2.10 [−0.31, +4.51] | no |
| 10 | MuLMS, held-out subfields | `crf_mr` votes vs `crf` votes | week 10, decision 2 | 7 | +2.71 [+0.22, +5.20] | passes |
| 11 | MuLMS, held-out subfields | `crf_mrs` vs `crf` | week 11, decision 1 | 7 | +2.04 [−0.85, +4.92] | no |
| 11 | MuLMS, held-out subfields | `crf_mrs` vs `crf_mr` | week 11, decision 2 | 7 | −0.06 [−1.09, +0.96] | no |
| 12 | MuLMS, held-out subfields | `crf_mrm_topic6` vs `crf` | week 12, decision 1 | 7 | +3.19 [−1.13, +7.51] | no |
| 12 | MuLMS, held-out subfields | `crf_mrm_topic6` vs `crf_mr` | week 12, decision 2 | 7 | +0.92 [−1.63, +3.48] | no |
| 12 | MaterioMiner fine-grained, unseen papers | `crf_mrm6` vs `crf` | week 12, the winner's analogue | 4 | +4.27 [+1.39, +7.15] | passes |
| 12 | MaterioMiner coarse-grained, unseen papers | `crf_mrm6` vs `crf` | week 12, the winner's analogue | 4 | +1.77 [+0.67, +2.88] | passes |
| 13 | MuLMS, held-out subfields | `crf_dc` vs `crf` | week 13, 13.1a | 7 | +0.82 [+0.18, +1.46] | passes |
| 13 | MuLMS, held-out subfields | `crf_mr_dc` vs `crf_mr` | week 13, 13.1b | 7 | +0.57 [−0.12, +1.25] | no |
| 13 | MaterioMiner fine-grained, unseen papers | `crf_mr_dc` vs `crf_mr` | week 13, 13.1d | 4 | +0.19 [−0.34, +0.73] | no |
| 13 | MaterioMiner coarse-grained, unseen papers | `crf_mr_dc` vs `crf_mr` | week 13, 13.1d | 4 | +0.11 [−0.08, +0.31] | no |
| 13 | SOFC-Exp, article folds | `crf_mr` vs `crf` | week 13, 13.3a | 5 | −0.48 [−1.36, +0.39] | no |
| 13 | SOFC-Exp, article folds | `crf_mt_mr` vs `crf` | week 13, 13.3b | 5 | −0.11 [−0.69, +0.46] | no |
| 13 | pooled: MuLMS, held-out subfields (7) + MaterioMiner fine-grained, unseen papers (4) | `crf_mrm_topic6` vs `crf` | week 13, 13.2 | 11 | +3.58 [+0.95, +6.21] | passes |
