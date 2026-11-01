Table 14. The unseen-name trade in full (ner_mentions.json, `chain`): per setting, each system against the one it builds on, the mean paired difference over the units in F1, recall and precision on the unseen and the seen test names and over all entities, with 95% CIs, on all ten seeds and, on MuLMS, on the decision seeds (300–900).

| setting | comparison | seeds | part | Δ F1 [95% CI] | Δ recall [95% CI] | Δ precision [95% CI] |
|---|---|---|---|---|---|---|
| MuLMS, 7 subfields | MR3 − CRF head | all ten | unseen names | +2.19 [−0.25, +4.63] | +3.76 [+0.73, +6.79] | −1.10 [−2.67, +0.47] |
| MuLMS, 7 subfields | MR3 − CRF head | all ten | seen names | +1.30 [−1.21, +3.81] | +5.06 [+1.90, +8.22] | −3.48 [−5.68, −1.29] |
| MuLMS, 7 subfields | MR3 − CRF head | all ten | all entities | +2.10 [−0.31, +4.51] | +3.89 [+0.91, +6.87] | −1.40 [−2.92, +0.13] |
| MuLMS, 7 subfields | MR3 − CRF head | decision seeds (300–900) | unseen names | +2.31 [−0.27, +4.89] | +4.05 [+0.94, +7.16] | −1.16 [−3.83, +1.51] |
| MuLMS, 7 subfields | MR3 − CRF head | decision seeds (300–900) | seen names | +1.83 [−0.76, +4.41] | +5.57 [+2.28, +8.85] | −2.86 [−5.42, −0.30] |
| MuLMS, 7 subfields | MR3 − CRF head | decision seeds (300–900) | all entities | +2.27 [−0.25, +4.79] | +4.21 [+1.13, +7.30] | −1.38 [−3.80, +1.04] |
| MuLMS, 7 subfields | MR6 − MR3 | all ten | unseen names | +0.22 [−0.45, +0.88] | +0.63 [−0.20, +1.46] | −0.26 [−1.31, +0.79] |
| MuLMS, 7 subfields | MR6 − MR3 | all ten | seen names | +0.18 [−0.98, +1.33] | +0.79 [−0.79, +2.37] | +0.08 [−1.58, +1.74] |
| MuLMS, 7 subfields | MR6 − MR3 | all ten | all entities | +0.23 [−0.39, +0.85] | +0.68 [−0.09, +1.44] | −0.22 [−1.21, +0.77] |
| MuLMS, 7 subfields | MR6 − MR3 | decision seeds (300–900) | unseen names | +0.43 [−0.32, +1.17] | +0.73 [+0.04, +1.42] | −0.05 [−2.04, +1.93] |
| MuLMS, 7 subfields | MR6 − MR3 | decision seeds (300–900) | seen names | +0.27 [−0.51, +1.04] | +0.52 [−0.61, +1.64] | +0.27 [−0.92, +1.46] |
| MuLMS, 7 subfields | MR6 − MR3 | decision seeds (300–900) | all entities | +0.41 [−0.35, +1.17] | +0.69 [+0.07, +1.30] | +0.01 [−1.85, +1.87] |
| MuLMS, 7 subfields | MR6 + names − MR6 | all ten | unseen names | +0.93 [−1.36, +3.22] | +3.24 [+1.36, +5.13] | −2.23 [−4.84, +0.38] |
| MuLMS, 7 subfields | MR6 + names − MR6 | all ten | seen names | +0.77 [−1.18, +2.72] | +2.43 [−1.04, +5.89] | −0.99 [−2.80, +0.81] |
| MuLMS, 7 subfields | MR6 + names − MR6 | all ten | all entities | +0.85 [−1.46, +3.17] | +3.12 [+1.23, +5.00] | −2.17 [−4.69, +0.35] |
| MuLMS, 7 subfields | MR6 + names − MR6 | decision seeds (300–900) | unseen names | +0.56 [−1.44, +2.56] | +2.56 [+0.76, +4.36] | −2.22 [−5.03, +0.60] |
| MuLMS, 7 subfields | MR6 + names − MR6 | decision seeds (300–900) | seen names | +0.47 [−1.39, +2.33] | +1.89 [−0.70, +4.48] | −0.96 [−3.61, +1.68] |
| MuLMS, 7 subfields | MR6 + names − MR6 | decision seeds (300–900) | all entities | +0.51 [−1.51, +2.53] | +2.48 [+0.67, +4.29] | −2.15 [−4.95, +0.66] |
| MuLMS, 7 subfields | MR6 + names (mixed corpus) − MR6 | all ten | unseen names | +0.64 [−1.33, +2.62] | +2.47 [−0.09, +5.02] | −1.85 [−3.56, −0.15] |
| MuLMS, 7 subfields | MR6 + names (mixed corpus) − MR6 | all ten | seen names | +0.99 [−0.70, +2.67] | +2.43 [−0.71, +5.58] | −0.02 [−2.48, +2.45] |
| MuLMS, 7 subfields | MR6 + names (mixed corpus) − MR6 | all ten | all entities | +0.70 [−1.19, +2.58] | +2.51 [+0.04, +4.98] | −1.68 [−3.35, −0.02] |
| MaterioMiner fine, 4 papers | MR3 − CRF head | all ten | unseen names | +3.19 [+1.66, +4.72] | +4.51 [+2.27, +6.74] | +0.35 [−2.01, +2.71] |
| MaterioMiner fine, 4 papers | MR3 − CRF head | all ten | seen names | +1.49 [−1.31, +4.30] | +3.38 [−0.02, +6.77] | −0.49 [−6.98, +6.00] |
| MaterioMiner fine, 4 papers | MR3 − CRF head | all ten | all entities | +2.77 [+1.51, +4.03] | +4.11 [+2.97, +5.25] | +0.32 [−3.23, +3.87] |
| MaterioMiner fine, 4 papers | MR6 − MR3 | all ten | unseen names | +0.46 [−2.16, +3.09] | +1.25 [−0.88, +3.38] | −0.49 [−4.41, +3.43] |
| MaterioMiner fine, 4 papers | MR6 − MR3 | all ten | seen names | +0.56 [−0.37, +1.49] | +1.21 [−1.26, +3.68] | +0.16 [−0.96, +1.29] |
| MaterioMiner fine, 4 papers | MR6 − MR3 | all ten | all entities | +0.20 [−1.47, +1.88] | +0.93 [+0.09, +1.77] | −0.57 [−3.79, +2.64] |
| MaterioMiner fine, 4 papers | MR6 + names − MR6 | all ten | unseen names | +1.26 [−3.40, +5.91] | +0.86 [−3.38, +5.10] | +1.32 [−6.62, +9.25] |
| MaterioMiner fine, 4 papers | MR6 + names − MR6 | all ten | seen names | +1.03 [−0.77, +2.82] | +0.77 [−1.43, +2.97] | +1.02 [−1.12, +3.16] |
| MaterioMiner fine, 4 papers | MR6 + names − MR6 | all ten | all entities | +1.29 [−2.51, +5.10] | +0.88 [−2.82, +4.58] | +1.56 [−5.05, +8.17] |
| MaterioMiner fine, 4 papers | multi-task + MR3 − multi-task | all ten | unseen names | +3.09 [+0.99, +5.18] | +3.46 [+1.66, +5.27] | +2.58 [−1.00, +6.17] |
| MaterioMiner fine, 4 papers | multi-task + MR3 − multi-task | all ten | seen names | +3.00 [−0.85, +6.85] | +5.34 [−4.38, +15.06] | +0.62 [−7.19, +8.44] |
| MaterioMiner fine, 4 papers | multi-task + MR3 − multi-task | all ten | all entities | +3.01 [+0.42, +5.60] | +3.40 [+0.67, +6.13] | +2.17 [−0.43, +4.78] |
| MaterioMiner fine, 4 papers | multi-task + MR6 − multi-task + MR3 | all ten | unseen names | −0.00 [−1.75, +1.74] | −0.42 [−3.49, +2.64] | +0.14 [−2.17, +2.46] |
| MaterioMiner fine, 4 papers | multi-task + MR6 − multi-task + MR3 | all ten | seen names | −1.13 [−4.03, +1.76] | −0.11 [−4.08, +3.86] | −1.78 [−5.15, +1.60] |
| MaterioMiner fine, 4 papers | multi-task + MR6 − multi-task + MR3 | all ten | all entities | −0.19 [−2.05, +1.67] | −0.33 [−3.43, +2.78] | −0.29 [−1.78, +1.19] |
| MaterioMiner fine, 4 papers | multi-task + MR6 + names − multi-task + MR6 | all ten | unseen names | +0.43 [−1.57, +2.44] | +0.58 [−2.93, +4.09] | +0.82 [−3.39, +5.03] |
| MaterioMiner fine, 4 papers | multi-task + MR6 + names − multi-task + MR6 | all ten | seen names | +2.43 [+0.40, +4.47] | +0.88 [−3.39, +5.16] | +3.39 [+2.65, +4.12] |
| MaterioMiner fine, 4 papers | multi-task + MR6 + names − multi-task + MR6 | all ten | all entities | +0.96 [−0.01, +1.92] | +0.85 [−2.03, +3.73] | +1.46 [−1.21, +4.13] |
| MaterioMiner coarse, 4 papers | MR3 − CRF head | all ten | unseen names | +1.58 [−0.02, +3.17] | +1.86 [+0.44, +3.27] | +1.30 [−0.53, +3.12] |
| MaterioMiner coarse, 4 papers | MR3 − CRF head | all ten | seen names | +0.12 [−0.83, +1.06] | +0.73 [−2.72, +4.18] | −0.31 [−2.60, +1.98] |
| MaterioMiner coarse, 4 papers | MR3 − CRF head | all ten | all entities | +1.28 [+0.06, +2.51] | +1.55 [+0.48, +2.62] | +1.02 [−0.61, +2.65] |
| MaterioMiner coarse, 4 papers | MR6 − MR3 | all ten | unseen names | −0.64 [−3.44, +2.15] | −0.27 [−1.99, +1.45] | −1.00 [−4.87, +2.87] |
| MaterioMiner coarse, 4 papers | MR6 − MR3 | all ten | seen names | +0.51 [−1.18, +2.21] | +2.98 [+0.65, +5.30] | −1.62 [−2.34, −0.91] |
| MaterioMiner coarse, 4 papers | MR6 − MR3 | all ten | all entities | −0.21 [−2.71, +2.29] | +0.45 [−1.38, +2.27] | −0.82 [−3.97, +2.33] |
| MaterioMiner coarse, 4 papers | MR6 + names − MR6 | all ten | unseen names | +1.37 [−1.75, +4.50] | +1.60 [−1.37, +4.57] | +1.16 [−2.24, +4.57] |
| MaterioMiner coarse, 4 papers | MR6 + names − MR6 | all ten | seen names | −0.48 [−3.19, +2.23] | −3.39 [−7.96, +1.18] | +1.90 [−0.56, +4.36] |
| MaterioMiner coarse, 4 papers | MR6 + names − MR6 | all ten | all entities | +0.70 [−1.60, +3.00] | +0.47 [−1.09, +2.03] | +0.90 [−2.08, +3.88] |
| MaterioMiner coarse, 4 papers | multi-task + MR3 − multi-task | all ten | unseen names | +2.02 [−0.19, +4.22] | +2.36 [+0.31, +4.42] | +1.68 [−0.68, +4.05] |
| MaterioMiner coarse, 4 papers | multi-task + MR3 − multi-task | all ten | seen names | +2.11 [−0.02, +4.24] | +4.50 [+0.12, +8.89] | −0.01 [−4.00, +3.97] |
| MaterioMiner coarse, 4 papers | multi-task + MR3 − multi-task | all ten | all entities | +2.30 [+0.49, +4.10] | +2.73 [+0.90, +4.56] | +1.87 [+0.00, +3.74] |
| MaterioMiner coarse, 4 papers | multi-task + MR6 − multi-task + MR3 | all ten | unseen names | +0.28 [−0.80, +1.37] | +0.65 [−0.73, +2.03] | −0.07 [−1.25, +1.11] |
| MaterioMiner coarse, 4 papers | multi-task + MR6 − multi-task + MR3 | all ten | seen names | −0.56 [−1.03, −0.08] | −0.27 [−1.92, +1.37] | −0.83 [−1.80, +0.14] |
| MaterioMiner coarse, 4 papers | multi-task + MR6 − multi-task + MR3 | all ten | all entities | +0.06 [−0.74, +0.86] | +0.41 [−0.69, +1.52] | −0.28 [−1.26, +0.70] |
| MaterioMiner coarse, 4 papers | multi-task + MR6 + names − multi-task + MR6 | all ten | unseen names | +1.20 [−0.82, +3.22] | +1.45 [−0.69, +3.58] | +0.97 [−1.09, +3.03] |
| MaterioMiner coarse, 4 papers | multi-task + MR6 + names − multi-task + MR6 | all ten | seen names | +0.02 [−1.30, +1.34] | −0.96 [−2.09, +0.16] | +0.83 [−1.27, +2.93] |
| MaterioMiner coarse, 4 papers | multi-task + MR6 + names − multi-task + MR6 | all ten | all entities | +0.85 [−0.72, +2.42] | +0.88 [−0.81, +2.58] | +0.82 [−0.64, +2.27] |
| SOFC-Exp, 5 folds | MR3 − CRF head | all ten | unseen names | −0.43 [−1.66, +0.79] | +0.08 [−0.63, +0.79] | −0.93 [−2.96, +1.10] |
| SOFC-Exp, 5 folds | MR3 − CRF head | all ten | seen names | −0.50 [−1.27, +0.28] | −0.45 [−1.98, +1.09] | −0.51 [−1.68, +0.66] |
| SOFC-Exp, 5 folds | MR3 − CRF head | all ten | all entities | −0.48 [−1.36, +0.39] | −0.22 [−0.99, +0.56] | −0.72 [−2.30, +0.87] |
