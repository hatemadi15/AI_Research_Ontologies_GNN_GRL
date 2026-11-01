Table 10. Recall on the unseen test names by lexicon membership (week 15, post hoc; ner_lexicon_recall.py from the saved predictions and the lexicons rebuilt by the names method's own teachers, batch w15a): per system, pooled over seeds and held-out units, recall on the unseen test names that the unit's lexicon holds (in) and on those it does not (out), and among the predicted entities whose name is unseen the share that are lexicon names and the precision of each part, with the F1 of each part from its pooled recall and precision (round 15); then, paired over the units, a method's recall gain over its reference on each part and the difference of the two gains with its 95% Student-t CI, and the F1 gain on the held names, each unit's F1 from its own pooled recall and precision on them (round 15). Units used: MuLMS, 7 held-out subfields (seeds 300–900): electrolysis, graphene, pemfc, poly, semi, sofc, steel; MaterioMiner fine-grained, 4 held-out papers (10 seeds): 10.1007_s11340-021-00758-x.conll, 10.1038_ncomms7164.conll, 10.1038_s41467-020-19071-7.conll, 10.3390_met10111458.conll; MaterioMiner coarse-grained, 4 held-out papers (10 seeds): 10.1007_s11340-021-00758-x.conll, 10.1038_ncomms7164.conll, 10.1038_s41467-020-19071-7.conll, 10.3390_met10111458.conll.

| setting | system | unseen test names in the lexicon (%) | recall, in | recall, out | unseen predictions in the lexicon (%) | precision, in | precision, out | F1, in | F1, out |
|---|---|---|---|---|---|---|---|---|---|
| MuLMS, 7 held-out subfields (seeds 300–900) | CRF head | 22.2 | 71.8 | 29.7 | 30.7 | 74.3 | 47.6 | 73.1 | 36.6 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR3 | 22.2 | 75.6 | 34.1 | 28.4 | 74.1 | 46.3 | 74.8 | 39.2 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 | 22.2 | 75.0 | 35.0 | 27.5 | 74.7 | 46.4 | 74.8 | 39.9 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names | 22.2 | 82.4 | 36.8 | 29.0 | 69.0 | 44.0 | 75.1 | 40.1 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names (mixed corpus) | 22.2 | 81.6 | 36.7 | 28.8 | 70.7 | 45.0 | 75.7 | 40.4 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | CRF head | 9.8 | 74.0 | 30.8 | 13.0 | 81.3 | 46.4 | 77.5 | 37.0 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR3 | 9.8 | 83.1 | 34.0 | 13.0 | 82.7 | 46.3 | 82.9 | 39.2 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 | 9.8 | 82.5 | 35.0 | 12.3 | 81.4 | 44.6 | 82.0 | 39.3 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 + names | 9.8 | 86.6 | 36.1 | 13.6 | 81.2 | 48.8 | 83.8 | 41.5 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | CRF head | 18.2 | 79.6 | 45.4 | 17.7 | 82.2 | 45.5 | 80.9 | 45.4 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR3 | 18.2 | 79.1 | 47.2 | 17.3 | 82.8 | 46.6 | 80.9 | 46.9 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 | 18.2 | 80.5 | 46.9 | 17.8 | 81.5 | 46.1 | 81.0 | 46.5 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 + names | 18.2 | 80.7 | 48.3 | 17.5 | 82.4 | 47.0 | 81.6 | 47.7 |


**Paired gains**

| setting | comparison | recall gain on the names in the lexicon [95% CI] | recall gain on the names not in it [95% CI] | in − out [95% CI] | units with in > out | F1 gain on the names in the lexicon [95% CI] |
|---|---|---|---|---|---|---|
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − MR6 | +7.29 [+3.27, +11.30] | +1.12 [−0.78, +3.02] | +6.17 [+2.44, +9.90] | 7/7 | +1.13 [−1.84, +4.11] |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − CRF head | +10.67 [+5.14, +16.20] | +6.34 [+2.60, +10.07] | +4.33 [+1.18, +7.48] | 6/7 | +3.29 [−1.27, +7.85] |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 − CRF head | +3.39 [+0.92, +5.85] | +5.22 [+1.92, +8.52] | −1.84 [−5.27, +1.60] | 3/7 | +2.15 [+0.15, +4.16] |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR3 − CRF head | +3.56 [+1.11, +6.01] | +4.22 [+0.87, +7.57] | −0.66 [−2.78, +1.45] | 3/7 | +2.02 [−0.02, +4.06] |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names (mixed corpus) − MR6 | +6.16 [+1.32, +11.01] | +1.31 [−1.07, +3.68] | +4.86 [+1.50, +8.21] | 7/7 | +1.18 [−1.63, +4.00] |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 + names − MR6 | +4.62 [−5.27, +14.50] | +0.29 [−4.23, +4.81] | +4.33 [−5.33, +13.98] | 3/4 | +1.96 [−6.54, +10.46] |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 + names − CRF head | +12.77 [−12.61, +38.15] | +6.00 [+2.84, +9.16] | +6.77 [−18.59, +32.13] | 2/4 | +6.57 [−18.80, +31.94] |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 − CRF head | +8.15 [−7.65, +23.95] | +5.71 [+1.05, +10.37] | +2.44 [−15.96, +20.85] | 2/4 | +4.61 [−12.46, +21.68] |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR3 − CRF head | +8.77 [−5.95, +23.49] | +4.17 [−0.17, +8.50] | +4.60 [−13.82, +23.03] | 3/4 | +5.73 [−9.03, +20.49] |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 + names − MR6 | +0.36 [−1.11, +1.84] | +1.91 [−1.46, +5.28] | −1.54 [−3.60, +0.51] | 0/4 | +1.17 [−2.23, +4.58] |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 + names − CRF head | +1.41 [−0.38, +3.19] | +3.62 [+1.18, +6.07] | −2.22 [−5.06, +0.62] | 0/4 | +1.47 [−1.36, +4.30] |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 − CRF head | +1.04 [−1.72, +3.80] | +1.72 [−0.37, +3.80] | −0.67 [−1.53, +0.19] | 0/4 | +0.30 [−2.84, +3.43] |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR3 − CRF head | −0.12 [−1.58, +1.34] | +2.32 [+0.88, +3.75] | −2.44 [−2.63, −2.24] | 0/4 | +0.62 [−1.51, +2.75] |
