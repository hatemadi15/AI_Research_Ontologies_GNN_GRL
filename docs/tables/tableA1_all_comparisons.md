Table A1. Every comparison on MaterioMiner: mean paired difference of micro F1 over splits or held-out papers, each side the mean over its seeds.

| week | machine | seeds | granularity | protocol | comparison | role | pairs | Δ micro F1 [95% CI] | CI above 0 |
|---|---|---|---|---|---|---|---|---|---|
| 3 | CPU | 3 | fine | random splits | `crf_mix3` vs `crf_ens3` | decision | 5 | +0.03 [−1.80, +1.86] | no |
| 3 | CPU | 3 | coarse | random splits | `crf_mix3` vs `crf_ens3` | decision | 5 | −0.87 [−2.09, +0.35] | no |
| 3 | CPU | 3 | fine | unseen papers | `crf_mix3` vs `crf_ens3` | reported | 4 | −1.34 [−4.76, +2.09] |  |
| 3 | CPU | 3 | coarse | unseen papers | `crf_mix3` vs `crf_ens3` | reported | 4 | −1.45 [−5.74, +2.84] |  |
| 3 | CPU | 3 | fine | random splits | `crf_mix3` vs `crf` | reported | 5 | +2.50 [+0.07, +4.92] |  |
| 3 | CPU | 3 | coarse | random splits | `crf_mix3` vs `crf` | reported | 5 | +0.93 [−0.52, +2.38] |  |
| 3 | CPU | 3 | fine | unseen papers | `crf_mix3` vs `crf` | reported | 4 | +0.71 [−3.77, +5.18] |  |
| 3 | CPU | 3 | coarse | unseen papers | `crf_mix3` vs `crf` | reported | 4 | +0.94 [−3.20, +5.07] |  |
| 3 | CPU | 3 | fine | random splits | `crf_ens3` vs `crf` | reported | 5 | +2.47 [+1.17, +3.77] |  |
| 3 | CPU | 3 | coarse | random splits | `crf_ens3` vs `crf` | reported | 5 | +1.80 [+0.91, +2.69] |  |
| 3 | CPU | 3 | fine | unseen papers | `crf_ens3` vs `crf` | reported | 4 | +2.04 [−0.23, +4.32] |  |
| 3 | CPU | 3 | coarse | unseen papers | `crf_ens3` vs `crf` | reported | 4 | +2.39 [+1.56, +3.21] |  |
| 3 | CPU | 3 | fine | random splits | `crf_scibert` vs `crf` | reported | 5 | −0.38 [−2.63, +1.87] |  |
| 3 | CPU | 3 | coarse | random splits | `crf_scibert` vs `crf` | reported | 5 | −0.99 [−1.71, −0.26] |  |
| 3 | CPU | 3 | fine | unseen papers | `crf_scibert` vs `crf` | reported | 4 | −4.83 [−6.66, −3.00] |  |
| 3 | CPU | 3 | coarse | unseen papers | `crf_scibert` vs `crf` | reported | 4 | −4.52 [−8.17, −0.87] |  |
| 3 | CPU | 3 | fine | random splits | `crf_pubmedbert` vs `crf` | reported | 5 | −1.42 [−5.21, +2.37] |  |
| 3 | CPU | 3 | coarse | random splits | `crf_pubmedbert` vs `crf` | reported | 5 | −2.99 [−4.67, −1.32] |  |
| 3 | CPU | 3 | fine | unseen papers | `crf_pubmedbert` vs `crf` | reported | 4 | −4.94 [−13.22, +3.34] |  |
| 3 | CPU | 3 | coarse | unseen papers | `crf_pubmedbert` vs `crf` | reported | 4 | −3.75 [−8.41, +0.90] |  |
| 3 | CPU | 3 | fine | random splits | `crf_ls` vs `crf` | reported | 5 | −0.58 [−3.41, +2.24] |  |
| 3 | CPU | 3 | coarse | random splits | `crf_ls` vs `crf` | reported | 5 | −0.77 [−1.74, +0.21] |  |
| 3 | CPU | 3 | fine | unseen papers | `crf_ls` vs `crf` | decision | 4 | −0.16 [−2.13, +1.80] | no |
| 3 | CPU | 3 | coarse | unseen papers | `crf_ls` vs `crf` | decision | 4 | +1.17 [+0.94, +1.41] | passes |
| 3 | CPU | 3 | fine | unseen papers | `crf_st` vs `crf` | decision | 4 | +2.91 [−0.73, +6.56] | no |
| 3 | CPU | 3 | coarse | unseen papers | `crf_st` vs `crf` | decision | 4 | −0.12 [−1.97, +1.73] | no |
| 3 | CPU | 3 | fine | random splits | `crf_mt` vs `crf` | decision | 5 | +1.87 [−2.25, +6.00] | no |
| 3 | CPU | 3 | coarse | random splits | `crf_mt` vs `crf` | decision | 5 | +0.75 [+0.10, +1.39] | passes |
| 3 | CPU | 3 | fine | unseen papers | `crf_mt` vs `crf` | reported | 4 | +3.39 [+0.45, +6.34] |  |
| 3 | CPU | 3 | coarse | unseen papers | `crf_mt` vs `crf` | reported | 4 | −0.60 [−4.55, +3.34] |  |
| 3 | CPU | 3 | fine | random splits | `crf_dapt2` vs `crf` | decision | 5 | +0.11 [−2.74, +2.97] | no |
| 3 | CPU | 3 | coarse | random splits | `crf_dapt2` vs `crf` | decision | 5 | −0.15 [−1.06, +0.76] | no |
| 3 | CPU | 3 | fine | unseen papers | `crf_dapt2` vs `crf` | reported | 4 | +0.14 [−2.60, +2.88] |  |
| 3 | CPU | 3 | coarse | unseen papers | `crf_dapt2` vs `crf` | reported | 4 | −1.54 [−4.34, +1.26] |  |
| 4 | CPU | 3 | fine | random splits | `crf_ls_ens3` vs `crf_ens3` | reported | 5 | −1.16 [−2.31, −0.01] |  |
| 4 | CPU | 3 | coarse | random splits | `crf_ls_ens3` vs `crf_ens3` | reported | 5 | −0.95 [−1.88, −0.02] |  |
| 4 | CPU | 3 | fine | unseen papers | `crf_ls_ens3` vs `crf_ens3` | decision | 4 | +0.36 [−2.53, +3.24] | no |
| 4 | CPU | 3 | coarse | unseen papers | `crf_ls_ens3` vs `crf_ens3` | decision | 4 | +1.63 [+0.11, +3.16] | passes |
| 4 | CPU | 3 | fine | unseen papers | `crf_st_ens3` vs `crf_ens3` | decision | 4 | +3.68 [−1.15, +8.52] | no |
| 4 | CPU | 3 | coarse | unseen papers | `crf_st_ens3` vs `crf_ens3` | decision | 4 | −0.09 [−3.52, +3.35] | no |
| 4 | CPU | 3 | fine | random splits | `crf_ls_ens3` vs `crf` | reported | 5 | +1.30 [−1.08, +3.68] |  |
| 4 | CPU | 3 | coarse | random splits | `crf_ls_ens3` vs `crf` | reported | 5 | +0.86 [−0.25, +1.96] |  |
| 4 | CPU | 3 | fine | unseen papers | `crf_ls_ens3` vs `crf` | reported | 4 | +2.40 [+0.30, +4.50] |  |
| 4 | CPU | 3 | coarse | unseen papers | `crf_ls_ens3` vs `crf` | reported | 4 | +4.02 [+3.18, +4.86] |  |
| 4 | CPU | 3 | fine | unseen papers | `crf_st_ens3` vs `crf` | reported | 4 | +5.72 [+2.33, +9.12] |  |
| 4 | CPU | 3 | coarse | unseen papers | `crf_st_ens3` vs `crf` | reported | 4 | +2.30 [−0.62, +5.21] |  |
| 4 | GPU | 3 | fine | random splits | `crf_mt_ens3` vs `crf_ens3` | decision | 5 | +1.26 [−1.08, +3.59] | no |
| 4 | GPU | 3 | coarse | random splits | `crf_mt_ens3` vs `crf_ens3` | decision | 5 | −0.07 [−2.02, +1.87] | no |
| 4 | GPU | 3 | fine | unseen papers | `crf_mt_ens3` vs `crf_ens3` | reported | 4 | +4.09 [−0.89, +9.07] |  |
| 4 | GPU | 3 | coarse | unseen papers | `crf_mt_ens3` vs `crf_ens3` | reported | 4 | −0.10 [−5.04, +4.85] |  |
| 4 | GPU | 3 | fine | random splits | `crf_mt_ls` vs `crf` | decision | 5 | +0.76 [−0.57, +2.10] | no |
| 4 | GPU | 3 | coarse | random splits | `crf_mt_ls` vs `crf` | decision | 5 | +0.65 [−0.63, +1.93] | no |
| 4 | GPU | 3 | fine | unseen papers | `crf_mt_ls` vs `crf` | decision | 4 | +4.08 [−1.99, +10.14] | no |
| 4 | GPU | 3 | coarse | unseen papers | `crf_mt_ls` vs `crf` | decision | 4 | −0.01 [−2.64, +2.63] | no |
| 4 | GPU | 3 | fine | random splits | `crf_st3` vs `crf` | reported | 5 | +1.22 [+0.60, +1.84] |  |
| 4 | GPU | 3 | coarse | random splits | `crf_st3` vs `crf` | reported | 5 | +0.71 [−0.96, +2.38] |  |
| 4 | GPU | 3 | fine | unseen papers | `crf_st3` vs `crf` | decision | 4 | +1.07 [+0.33, +1.80] | passes |
| 4 | GPU | 3 | coarse | unseen papers | `crf_st3` vs `crf` | decision | 4 | +0.85 [−0.58, +2.29] | no |
| 4 | GPU | 3 | fine | random splits | `crf_ls_ens3` vs `crf_ens3` | reported | 5 | −1.74 [−3.84, +0.35] |  |
| 4 | GPU | 3 | coarse | random splits | `crf_ls_ens3` vs `crf_ens3` | reported | 5 | −0.60 [−2.98, +1.78] |  |
| 4 | GPU | 3 | fine | unseen papers | `crf_ls_ens3` vs `crf_ens3` | reported | 4 | +0.51 [−1.56, +2.59] |  |
| 4 | GPU | 3 | coarse | unseen papers | `crf_ls_ens3` vs `crf_ens3` | reported | 4 | +0.40 [−1.36, +2.16] |  |
| 4 | GPU | 3 | fine | unseen papers | `crf_st_ens3` vs `crf_ens3` | reported | 4 | +2.40 [−2.53, +7.33] |  |
| 4 | GPU | 3 | coarse | unseen papers | `crf_st_ens3` vs `crf_ens3` | reported | 4 | −0.09 [−2.85, +2.67] |  |
| 4 | GPU | 3 | fine | random splits | `crf_ens3` vs `crf` | reported | 5 | +2.30 [+1.75, +2.85] |  |
| 4 | GPU | 3 | coarse | random splits | `crf_ens3` vs `crf` | reported | 5 | +1.61 [+0.65, +2.57] |  |
| 4 | GPU | 3 | fine | unseen papers | `crf_ens3` vs `crf` | reported | 4 | +2.08 [+0.97, +3.19] |  |
| 4 | GPU | 3 | coarse | unseen papers | `crf_ens3` vs `crf` | reported | 4 | +2.02 [−0.03, +4.08] |  |
| 4 | GPU | 3 | fine | random splits | `crf_mt_ens3` vs `crf` | reported | 5 | +3.56 [+1.39, +5.72] |  |
| 4 | GPU | 3 | coarse | random splits | `crf_mt_ens3` vs `crf` | reported | 5 | +1.54 [−0.57, +3.64] |  |
| 4 | GPU | 3 | fine | unseen papers | `crf_mt_ens3` vs `crf` | reported | 4 | +6.18 [+1.51, +10.84] |  |
| 4 | GPU | 3 | coarse | unseen papers | `crf_mt_ens3` vs `crf` | reported | 4 | +1.93 [−1.68, +5.53] |  |
| 4 | GPU | 3 | fine | random splits | `crf_mt` vs `crf` | reported | 5 | +1.74 [−0.36, +3.84] |  |
| 4 | GPU | 3 | coarse | random splits | `crf_mt` vs `crf` | reported | 5 | −0.08 [−1.71, +1.55] |  |
| 4 | GPU | 3 | fine | unseen papers | `crf_mt` vs `crf` | reported | 4 | +3.87 [−2.25, +9.99] |  |
| 4 | GPU | 3 | coarse | unseen papers | `crf_mt` vs `crf` | reported | 4 | −0.68 [−4.51, +3.15] |  |
| 4 | GPU | 3 | fine | random splits | `crf_ls` vs `crf` | reported | 5 | −0.76 [−3.61, +2.09] |  |
| 4 | GPU | 3 | coarse | random splits | `crf_ls` vs `crf` | reported | 5 | −0.15 [−1.68, +1.38] |  |
| 4 | GPU | 3 | fine | unseen papers | `crf_ls` vs `crf` | reported | 4 | +1.32 [−0.43, +3.06] |  |
| 4 | GPU | 3 | coarse | unseen papers | `crf_ls` vs `crf` | reported | 4 | +0.07 [−2.47, +2.61] |  |
| 4 | GPU | 3 | fine | unseen papers | `crf_st` vs `crf` | reported | 4 | +2.14 [−1.02, +5.29] |  |
| 4 | GPU | 3 | coarse | unseen papers | `crf_st` vs `crf` | reported | 4 | −1.09 [−4.31, +2.14] |  |
| 4 | GPU | 3 | fine | random splits | `crf_mt_ls` vs `crf_mt` | reported | 5 | −0.98 [−2.24, +0.28] |  |
| 4 | GPU | 3 | coarse | random splits | `crf_mt_ls` vs `crf_mt` | reported | 5 | +0.73 [−0.22, +1.68] |  |
| 4 | GPU | 3 | fine | unseen papers | `crf_mt_ls` vs `crf_mt` | reported | 4 | +0.21 [−5.83, +6.24] |  |
| 4 | GPU | 3 | coarse | unseen papers | `crf_mt_ls` vs `crf_mt` | reported | 4 | +0.67 [−0.60, +1.95] |  |
| 4 | GPU | 3 | fine | random splits | `crf_mt_ls` vs `crf_ls` | reported | 5 | +1.52 [−1.34, +4.38] |  |
| 4 | GPU | 3 | coarse | random splits | `crf_mt_ls` vs `crf_ls` | reported | 5 | +0.80 [−0.74, +2.34] |  |
| 4 | GPU | 3 | fine | unseen papers | `crf_mt_ls` vs `crf_ls` | reported | 4 | +2.76 [−1.63, +7.14] |  |
| 4 | GPU | 3 | coarse | unseen papers | `crf_mt_ls` vs `crf_ls` | reported | 4 | −0.07 [−3.50, +3.35] |  |
| 4 | GPU | 3 | fine | unseen papers | `crf_st3` vs `crf_st` | reported | 4 | −1.07 [−4.74, +2.60] |  |
| 4 | GPU | 3 | coarse | unseen papers | `crf_st3` vs `crf_st` | reported | 4 | +1.94 [−2.68, +6.56] |  |
| 4 | GPU | 3 | fine | random splits | `crf_st3` vs `crf_ens3` | reported | 5 | −1.08 [−2.14, −0.03] |  |
| 4 | GPU | 3 | coarse | random splits | `crf_st3` vs `crf_ens3` | reported | 5 | −0.90 [−3.20, +1.39] |  |
| 4 | GPU | 3 | fine | unseen papers | `crf_st3` vs `crf_ens3` | reported | 4 | −1.01 [−1.63, −0.39] |  |
| 4 | GPU | 3 | coarse | unseen papers | `crf_st3` vs `crf_ens3` | reported | 4 | −1.17 [−2.61, +0.27] |  |
| 4 | GPU | 3 | fine | random splits | `crf` vs `crf` | descriptive | 5 | +0.31 [−2.38, +3.00] |  |
| 4 | GPU | 3 | coarse | random splits | `crf` vs `crf` | descriptive | 5 | −0.07 [−1.19, +1.05] |  |
| 4 | GPU | 3 | fine | unseen papers | `crf` vs `crf` | descriptive | 4 | −1.13 [−3.07, +0.81] |  |
| 4 | GPU | 3 | coarse | unseen papers | `crf` vs `crf` | descriptive | 4 | +0.24 [−1.52, +1.99] |  |
| 4 | GPU | 3 | fine | random splits | `crf_ens3` vs `crf_ens3` | descriptive | 5 | +0.14 [−1.79, +2.08] |  |
| 4 | GPU | 3 | coarse | random splits | `crf_ens3` vs `crf_ens3` | descriptive | 5 | −0.26 [−1.62, +1.10] |  |
| 4 | GPU | 3 | fine | unseen papers | `crf_ens3` vs `crf_ens3` | descriptive | 4 | −1.09 [−3.09, +0.90] |  |
| 4 | GPU | 3 | coarse | unseen papers | `crf_ens3` vs `crf_ens3` | descriptive | 4 | −0.12 [−0.70, +0.45] |  |
| 4 | GPU | 3 | fine | random splits | `crf_mt` vs `crf_mt` | descriptive | 5 | +0.18 [−2.07, +2.43] |  |
| 4 | GPU | 3 | coarse | random splits | `crf_mt` vs `crf_mt` | descriptive | 5 | −0.89 [−1.88, +0.09] |  |
| 4 | GPU | 3 | fine | unseen papers | `crf_mt` vs `crf_mt` | descriptive | 4 | −0.66 [−4.34, +3.02] |  |
| 4 | GPU | 3 | coarse | unseen papers | `crf_mt` vs `crf_mt` | descriptive | 4 | +0.16 [−1.02, +1.34] |  |
| 4 | GPU | 3 | fine | random splits | `crf_ls` vs `crf_ls` | descriptive | 5 | +0.14 [−3.46, +3.73] |  |
| 4 | GPU | 3 | coarse | random splits | `crf_ls` vs `crf_ls` | descriptive | 5 | +0.55 [−0.50, +1.60] |  |
| 4 | GPU | 3 | fine | unseen papers | `crf_ls` vs `crf_ls` | descriptive | 4 | +0.35 [−1.18, +1.88] |  |
| 4 | GPU | 3 | coarse | unseen papers | `crf_ls` vs `crf_ls` | descriptive | 4 | −0.87 [−3.73, +1.99] |  |
| 4 | GPU | 3 | fine | unseen papers | `crf_st` vs `crf_st` | descriptive | 4 | −1.91 [−4.19, +0.37] |  |
| 4 | GPU | 3 | coarse | unseen papers | `crf_st` vs `crf_st` | descriptive | 4 | −0.73 [−3.37, +1.91] |  |
| 4 | GPU | 3 | fine | random splits | `crf_ls_ens3` vs `crf_ls_ens3` | descriptive | 5 | −0.44 [−3.72, +2.85] |  |
| 4 | GPU | 3 | coarse | random splits | `crf_ls_ens3` vs `crf_ls_ens3` | descriptive | 5 | +0.08 [−0.82, +0.98] |  |
| 4 | GPU | 3 | fine | unseen papers | `crf_ls_ens3` vs `crf_ls_ens3` | descriptive | 4 | −0.94 [−3.15, +1.28] |  |
| 4 | GPU | 3 | coarse | unseen papers | `crf_ls_ens3` vs `crf_ls_ens3` | descriptive | 4 | −1.36 [−2.57, −0.14] |  |
| 4 | GPU | 3 | fine | unseen papers | `crf_st_ens3` vs `crf_st_ens3` | descriptive | 4 | −2.38 [−4.71, −0.04] |  |
| 4 | GPU | 3 | coarse | unseen papers | `crf_st_ens3` vs `crf_st_ens3` | descriptive | 4 | −0.13 [−1.41, +1.16] |  |
| 5 | GPU | 10 | fine | random splits | `crf_mt` vs `crf` | decision | 5 | +1.83 [+0.32, +3.34] | passes |
| 5 | GPU | 10 | coarse | random splits | `crf_mt` vs `crf` | decision | 5 | +0.71 [−0.15, +1.56] | no |
| 5 | GPU | 10 | fine | unseen papers | `crf_mt` vs `crf` | reported | 4 | +3.72 [+0.57, +6.87] |  |
| 5 | GPU | 10 | coarse | unseen papers | `crf_mt` vs `crf` | reported | 4 | −0.56 [−1.91, +0.79] |  |
| 5 | GPU | 10 | fine | random splits | `crf_ls` vs `crf` | reported | 5 | −0.45 [−1.63, +0.73] |  |
| 5 | GPU | 10 | coarse | random splits | `crf_ls` vs `crf` | reported | 5 | +0.42 [−1.19, +2.04] |  |
| 5 | GPU | 10 | fine | unseen papers | `crf_ls` vs `crf` | decision | 4 | +1.23 [−0.96, +3.42] | no |
| 5 | GPU | 10 | coarse | unseen papers | `crf_ls` vs `crf` | decision | 4 | −0.36 [−1.66, +0.94] | no |
| 5 | GPU | 10 | fine | random splits | `crf_st3` vs `crf` | reported | 5 | +1.05 [+0.10, +2.00] |  |
| 5 | GPU | 10 | coarse | random splits | `crf_st3` vs `crf` | reported | 5 | +1.04 [+0.16, +1.91] |  |
| 5 | GPU | 10 | fine | unseen papers | `crf_st3` vs `crf` | decision | 4 | +1.74 [−1.16, +4.64] | no |
| 5 | GPU | 10 | coarse | unseen papers | `crf_st3` vs `crf` | decision | 4 | +1.56 [−0.88, +4.00] | no |
| 5 | GPU | 10 | fine | random splits | `crf_mt_ens3` vs `crf_ens3` | decision | 5 | +1.14 [−0.98, +3.27] | no |
| 5 | GPU | 10 | coarse | random splits | `crf_mt_ens3` vs `crf_ens3` | decision | 5 | +0.79 [+0.17, +1.42] | passes |
| 5 | GPU | 10 | fine | unseen papers | `crf_mt_ens3` vs `crf_ens3` | decision | 4 | +3.65 [+0.49, +6.82] | passes |
| 5 | GPU | 10 | coarse | unseen papers | `crf_mt_ens3` vs `crf_ens3` | decision | 4 | −0.32 [−1.07, +0.44] | no |
| 5 | GPU | 10 | fine | random splits | `crf_ls_ens3` vs `crf_ens3` | decision | 5 | −0.79 [−2.02, +0.43] | no |
| 5 | GPU | 10 | coarse | random splits | `crf_ls_ens3` vs `crf_ens3` | decision | 5 | +0.31 [−1.27, +1.89] | no |
| 5 | GPU | 10 | fine | unseen papers | `crf_ls_ens3` vs `crf_ens3` | decision | 4 | +0.57 [−1.46, +2.60] | no |
| 5 | GPU | 10 | coarse | unseen papers | `crf_ls_ens3` vs `crf_ens3` | decision | 4 | −0.63 [−1.72, +0.46] | no |
| 5 | GPU | 10 | fine | random splits | `crf_st3_ens3` vs `crf_ens3` | decision | 5 | +0.98 [−0.15, +2.10] | no |
| 5 | GPU | 10 | coarse | random splits | `crf_st3_ens3` vs `crf_ens3` | decision | 5 | +1.12 [−0.36, +2.60] | no |
| 5 | GPU | 10 | fine | unseen papers | `crf_st3_ens3` vs `crf_ens3` | decision | 4 | +1.12 [−2.71, +4.96] | no |
| 5 | GPU | 10 | coarse | unseen papers | `crf_st3_ens3` vs `crf_ens3` | decision | 4 | +0.93 [−1.66, +3.53] | no |
| 5 | GPU | 10 | fine | random splits | `crf_ens3` vs `crf` | reported | 5 | +2.20 [+1.61, +2.78] |  |
| 5 | GPU | 10 | coarse | random splits | `crf_ens3` vs `crf` | reported | 5 | +1.67 [+1.02, +2.32] |  |
| 5 | GPU | 10 | fine | unseen papers | `crf_ens3` vs `crf` | reported | 4 | +2.36 [+1.84, +2.88] |  |
| 5 | GPU | 10 | coarse | unseen papers | `crf_ens3` vs `crf` | reported | 4 | +2.50 [+1.86, +3.13] |  |
| 6 | GPU | 3 | fine | unseen papers | `crf_deberta` vs `crf` | screen | 4 | −4.38 [−7.32, −1.45] | no |
| 6 | GPU | 3 | coarse | unseen papers | `crf_deberta` vs `crf` | screen | 4 | −8.01 [−11.72, −4.30] | no |
| 6 | GPU | 3 | fine | unseen papers | `crf_ctx` vs `crf` | screen | 4 | +2.66 [+0.91, +4.40] | passes |
| 6 | GPU | 3 | coarse | unseen papers | `crf_ctx` vs `crf` | screen | 4 | −0.50 [−3.82, +2.83] | no |
| 6 | GPU | 3 | fine | unseen papers | `crf_mt_st3` vs `crf` | screen | 4 | +6.06 [+1.05, +11.06] | passes |
| 6 | GPU | 3 | coarse | unseen papers | `crf_mt_st3` vs `crf` | screen | 4 | +0.25 [−1.12, +1.62] | no |
| 6 | GPU | 3 | fine | unseen papers | `crf_mt_st3` vs `crf_mt` | reported | 4 | +2.19 [−1.61, +5.98] |  |
| 6 | GPU | 3 | coarse | unseen papers | `crf_mt_st3` vs `crf_mt` | reported | 4 | +0.93 [−2.82, +4.68] |  |
| 6 | GPU | 3 | fine | unseen papers | `crf_deberta_ens3` vs `crf_ens3` | reported | 4 | −2.17 [−5.12, +0.78] |  |
| 6 | GPU | 3 | coarse | unseen papers | `crf_deberta_ens3` vs `crf_ens3` | reported | 4 | −6.47 [−10.54, −2.39] |  |
| 6 | GPU | 3 | fine | unseen papers | `crf_ctx_ens3` vs `crf_ens3` | reported | 4 | +2.92 [+0.02, +5.83] |  |
| 6 | GPU | 3 | coarse | unseen papers | `crf_ctx_ens3` vs `crf_ens3` | reported | 4 | +0.29 [−4.64, +5.22] |  |
| 6 | GPU | 3 | fine | unseen papers | `crf_mt_st3_ens3` vs `crf_ens3` | reported | 4 | +5.73 [+0.70, +10.77] |  |
| 6 | GPU | 3 | coarse | unseen papers | `crf_mt_st3_ens3` vs `crf_ens3` | reported | 4 | +0.24 [−1.45, +1.93] |  |
| 6 | GPU | 10 | fine | unseen papers | `crf_ctx` vs `crf` | decision | 4 | +1.14 [−0.94, +3.22] | no |
| 6 | GPU | 10 | coarse | unseen papers | `crf_ctx` vs `crf` | decision | 4 | +0.12 [−2.03, +2.27] | no |
| 6 | GPU | 10 | fine | unseen papers | `crf_ctx_ens3` vs `crf_mt_ens3` | decision | 4 | −1.44 [−3.80, +0.91] | no |
| 6 | GPU | 10 | coarse | unseen papers | `crf_ctx_ens3` vs `crf_mt_ens3` | reported | 4 | +1.07 [−1.23, +3.38] |  |
| 6 | GPU | 10 | fine | unseen papers | `crf_ctx_ens3` vs `crf_ens3` | reported | 4 | +2.21 [+0.01, +4.41] |  |
| 6 | GPU | 10 | coarse | unseen papers | `crf_ctx_ens3` vs `crf_ens3` | decision | 4 | +0.76 [−1.79, +3.31] | no |
| 7 | GPU | 3 | fine | unseen papers | `crf_mr` vs `crf` | screen | 4 | +3.25 [−2.33, +8.82] | no |
| 7 | GPU | 3 | coarse | unseen papers | `crf_mr` vs `crf` | screen | 4 | +0.25 [−2.21, +2.72] | no |
| 7 | GPU | 3 | fine | unseen papers | `crf_mr_ens3` vs `crf_ens3` | reported | 4 | +5.00 [−0.50, +10.51] |  |
| 7 | GPU | 3 | coarse | unseen papers | `crf_mr_ens3` vs `crf_ens3` | reported | 4 | +1.16 [−2.11, +4.43] |  |
| 7 | GPU | 10 | fine | random splits | `crf_dc` vs `crf` | reported | 5 | −0.12 [−0.40, +0.16] |  |
| 7 | GPU | 10 | coarse | random splits | `crf_dc` vs `crf` | reported | 5 | −0.02 [−0.06, +0.02] |  |
| 7 | GPU | 10 | fine | unseen papers | `crf_dc` vs `crf` | decision | 4 | +0.30 [−0.35, +0.95] | no |
| 7 | GPU | 10 | coarse | unseen papers | `crf_dc` vs `crf` | decision | 4 | +0.12 [+0.01, +0.24] | passes |
| 7 | GPU | 10 | fine | unseen papers | `crf_mt_ens3_dc` vs `crf_mt_ens3` | decision | 4 | +0.33 [−0.74, +1.40] | no |
| 7 | GPU | 10 | coarse | unseen papers | `crf_mt_ens3_dc` vs `crf_mt_ens3` | reported | 4 | +0.17 [−0.22, +0.57] |  |
| 7 | GPU | 10 | fine | unseen papers | `crf_ens3_dc` vs `crf_ens3` | reported | 4 | +0.62 [−0.64, +1.89] |  |
| 7 | GPU | 10 | coarse | unseen papers | `crf_ens3_dc` vs `crf_ens3` | decision | 4 | +0.20 [−0.15, +0.55] | no |
| 7 | GPU | 10 | fine | random splits | `crf_dcr` vs `crf` | reported | 5 | −0.16 [−0.32, −0.01] |  |
| 7 | GPU | 10 | coarse | random splits | `crf_dcr` vs `crf` | reported | 5 | −0.11 [−0.26, +0.03] |  |
| 7 | GPU | 10 | fine | unseen papers | `crf_dcr` vs `crf` | reported | 4 | +0.31 [−0.73, +1.35] |  |
| 7 | GPU | 10 | coarse | unseen papers | `crf_dcr` vs `crf` | reported | 4 | +0.64 [+0.29, +1.00] |  |
| 7 | GPU | 10 | fine | unseen papers | `crf_mt_ens3_dcr` vs `crf_mt_ens3` | reported | 4 | +0.11 [−1.62, +1.84] |  |
| 7 | GPU | 10 | coarse | unseen papers | `crf_mt_ens3_dcr` vs `crf_mt_ens3` | reported | 4 | +0.60 [−0.20, +1.39] |  |
| 7 | GPU | 10 | fine | unseen papers | `crf_ens3_dcr` vs `crf_ens3` | reported | 4 | +0.56 [−2.03, +3.16] |  |
| 7 | GPU | 10 | coarse | unseen papers | `crf_ens3_dcr` vs `crf_ens3` | reported | 4 | +0.83 [+0.32, +1.34] |  |
| 7 | GPU | 10 | fine | unseen papers | `crf_ctx_dc` vs `crf_ctx` | reported | 4 | +0.10 [−0.67, +0.88] |  |
| 7 | GPU | 10 | coarse | unseen papers | `crf_ctx_dc` vs `crf_ctx` | reported | 4 | +0.06 [−0.09, +0.21] |  |
| 7 | GPU | 10 | fine | unseen papers | `crf_ctx_ens3_dc` vs `crf_ctx_ens3` | reported | 4 | +0.45 [−0.46, +1.36] |  |
| 7 | GPU | 10 | coarse | unseen papers | `crf_ctx_ens3_dc` vs `crf_ctx_ens3` | reported | 4 | +0.04 [−0.19, +0.28] |  |
| 7 | GPU | 10 | fine | unseen papers | `crf_eb` vs `crf` | decision | 4 | −0.59 [−1.75, +0.57] | no |
| 7 | GPU | 10 | coarse | unseen papers | `crf_eb` vs `crf` | decision | 4 | +0.17 [−0.14, +0.48] | no |
| 7 | GPU | 10 | fine | unseen papers | `crf_eb_ens3` vs `crf_mt_ens3` | decision | 4 | −3.53 [−8.10, +1.05] | no |
| 7 | GPU | 10 | coarse | unseen papers | `crf_eb_ens3` vs `crf_mt_ens3` | reported | 4 | +0.23 [−1.03, +1.49] |  |
| 7 | GPU | 10 | fine | unseen papers | `crf_eb_ens3` vs `crf_ens3` | reported | 4 | +0.13 [−2.20, +2.46] |  |
| 7 | GPU | 10 | coarse | unseen papers | `crf_eb_ens3` vs `crf_ens3` | decision | 4 | −0.09 [−0.75, +0.58] | no |
| 7 | GPU | 10 | fine | unseen papers | `crf_ebv` vs `crf` | reported | 4 | +0.32 [−0.83, +1.46] |  |
| 7 | GPU | 10 | coarse | unseen papers | `crf_ebv` vs `crf` | reported | 4 | +0.09 [−0.24, +0.43] |  |
| 7 | GPU | 10 | fine | unseen papers | `crf_cf` vs `crf` | decision | 4 | +0.34 [−0.91, +1.59] | no |
| 7 | GPU | 10 | fine | unseen papers | `crf_mt_ens3_cf` vs `crf_mt_ens3` | decision | 4 | +0.12 [−1.76, +2.01] | no |
| 7 | GPU | 10 | fine | unseen papers | `crf_mt_cf` vs `crf_mt` | reported | 4 | +0.07 [−1.40, +1.54] |  |
| 7 | GPU | 10 | fine | unseen papers | `crf_mr` vs `crf` | decision | 4 | +2.77 [+1.51, +4.03] | passes |
| 7 | GPU | 10 | coarse | unseen papers | `crf_mr` vs `crf` | decision | 4 | +1.28 [+0.06, +2.51] | passes |
| 7 | GPU | 10 | fine | unseen papers | `crf_mr_ens3` vs `crf_mt_ens3` | decision | 4 | −0.04 [−2.09, +2.01] | no |
| 7 | GPU | 10 | coarse | unseen papers | `crf_mr_ens3` vs `crf_mt_ens3` | reported | 4 | +2.57 [−0.19, +5.34] |  |
| 7 | GPU | 10 | fine | unseen papers | `crf_mr_ens3` vs `crf_ens3` | reported | 4 | +3.61 [+1.71, +5.52] |  |
| 7 | GPU | 10 | coarse | unseen papers | `crf_mr_ens3` vs `crf_ens3` | decision | 4 | +2.26 [−0.12, +4.64] | no |
