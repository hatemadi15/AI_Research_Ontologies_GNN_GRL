Table 7. The ablations of the names method (week 14, 14.3; reported, no decision): on MuLMS's 7 held-out subfields at the screen's seed offsets 0 to 200, the method with one setting changed, each against the method, mention replacement with 3 copies and the CRF head on the same seeds (mean paired micro-F1 difference over the subfields with its 95% CI); the distinct names the three teachers admit, summed over the subfields, and the share of the held-out subfields' unseen test names among them.

| setting (the method's value) | units | distinct silver names | unseen test names among them (%) | micro F1 | Δ vs the method [95% CI] | Δ vs mention replacement [95% CI] | Δ vs CRF head [95% CI] |
|---|---|---|---|---|---|---|---|
| the method: 40,000 sentences, threshold 0.9, 3 of 3 teachers, a uniform pool | 7 | 54,095 | 22.2 | 52.09 |  | +1.45 [−1.35, +4.25] | +3.16 [−1.96, +8.28] |
| the corpus's first 10,000 sentences (40,000) | 7 | 19,171 | 16.4 | 51.94 | −0.15 [−1.78, +1.48] | +1.30 [−1.14, +3.74] | +3.01 [−1.21, +7.23] |
| the corpus's first 20,000 sentences (40,000) | 7 | 32,461 | 20.1 | 51.93 | −0.16 [−2.45, +2.14] | +1.30 [−0.56, +3.15] | +3.00 [−0.49, +6.49] |
| marginal threshold 0.7 (0.9) | 7 | 77,879 | 31.5 | 51.93 | −0.16 [−1.62, +1.31] | +1.29 [−0.95, +3.53] | +3.00 [−1.85, +7.85] |
| 2 of 3 teachers agree (3 of 3) | 7 | 78,604 | 28.8 | 52.20 | +0.11 [−1.58, +1.81] | +1.56 [+0.07, +3.05] | +3.27 [−0.85, +7.38] |
| harvested names in half of the draws (a uniform pool) | 7 | 54,095 | 22.2 | 52.18 | +0.09 [−1.63, +1.80] | +1.54 [−0.74, +3.82] | +3.24 [−0.73, +7.22] |
