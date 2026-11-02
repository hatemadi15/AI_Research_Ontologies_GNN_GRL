Table 11. The harvested names on the strongest MaterioMiner base (week 17, post review; the multi-task model with 6-copy mention replacement, with and without the fatigue corpus's names; 4 held-out papers, 10 seeds, both granularities from one model): micro F1 with its 95% CI over the papers, the names against the matched control (decisions 17.1 and 17.2, the week-6 rule), and both against multi-task + MR3, the multi-task model and the CRF head (reported).

| granularity | system | micro F1 [95% CI], 4 papers | Δ vs the control [95% CI] | Δ vs multi-task + MR3 | Δ vs the multi-task model | Δ vs the CRF head |
|---|---|---|---|---|---|---|
| fine-grained | multi-task + MR6 + names | 53.95 [47.52, 60.38] | +0.96 [−0.01, +1.92] (decision 17.1, no) | +0.76 [−0.15, +1.68] | +3.77 [+1.34, +6.21] | +7.49 [+5.96, +9.02] |
| fine-grained | multi-task + MR6 (the control) | 52.99 [46.93, 59.06] |  | −0.19 [−2.05, +1.67] | +2.82 [−0.00, +5.64] | +6.54 [+4.14, +8.94] |
| coarse-grained | multi-task + MR6 + names | 61.25 [53.71, 68.80] | +0.85 [−0.72, +2.42] (decision 17.2, no) | +0.91 [−0.14, +1.97] | +3.21 [+0.46, +5.96] | +2.65 [−0.09, +5.38] |
| coarse-grained | multi-task + MR6 (the control) | 60.40 [53.66, 67.15] |  | +0.06 [−0.74, +0.86] | +2.36 [+1.15, +3.56] | +1.80 [+0.19, +3.40] |
