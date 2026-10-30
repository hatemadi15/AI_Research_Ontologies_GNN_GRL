Table 5. The CPU probe (no decision): per held-out MuLMS subfield, the share of the unseen test names that the topic corpus holds and that the two silver filters admit at threshold 0.9; the held-out CRF head's precision and recall (10 model seeds); and the change of its micro F1 when every verbatim match of a lexicon name is tagged where it predicted O (fill), for each 0.9 lexicon and for the gold unseen names themselves, which also override its tags in the last column.

| subfield | corpus holds (%) | sentence filter admits (%) | mention filter admits (%) | CRF precision | CRF recall | Δ F1, sentence lexicon | Δ F1, mention lexicon | Δ F1, gold names (fill) | Δ F1, gold names (override) |
|---|---|---|---|---|---|---|---|---|---|
| electrolysis | 70.7 | 12.5 | 29.9 | 57.3 | 54.7 | +0.85 | −0.77 | −6.85 | +4.47 |
| graphene | 64.0 | 15.3 | 28.8 | 56.8 | 44.9 | −2.86 | −5.07 | +9.11 | +24.32 |
| pemfc | 73.9 | 19.2 | 29.4 | 60.5 | 42.9 | +3.16 | +3.97 | −10.51 | −5.91 |
| poly | 65.9 | 8.0 | 18.3 | 55.9 | 48.7 | −0.25 | −3.31 | −3.63 | +2.22 |
| semi | 69.3 | 4.6 | 9.4 | 54.0 | 23.5 | +1.20 | +2.27 | +17.29 | +22.00 |
| sofc | 80.8 | 25.8 | 34.4 | 57.5 | 35.3 | +5.79 | +5.91 | +15.16 | +27.06 |
| steel | 55.0 | 13.3 | 25.0 | 64.4 | 50.9 | +0.01 | −10.03 | −1.82 | +4.13 |
| mean | 68.5 | 14.1 | 25.0 | 58.1 | 43.0 | +1.13 | −1.00 | +2.68 | +11.18 |
