Table 4. The CRF head (10 model seeds): the share of gold test entities whose words are no entity of their type in the training sentences, recall on seen and unseen names (pooled), and F1 under the matching criteria of SemEval-2013 Task 9.1 (mean over splits, papers, subfields or seeds).

| setting | unseen names (%) | recall, seen | recall, unseen | F1, strict | F1, exact | F1, type | F1, overlap |
|---|---|---|---|---|---|---|---|
| fine, random splits | 34.5 | 88.2 | 59.4 | 77.49 | 81.09 | 83.81 | 88.83 |
| fine, unseen papers | 81.2 | 71.4 | 35.0 | 46.46 | 56.10 | 56.36 | 70.58 |
| coarse, random splits | 39.5 | 89.0 | 63.0 | 76.96 | 83.91 | 84.46 | 93.38 |
| coarse, unseen papers | 78.8 | 82.1 | 51.6 | 58.60 | 74.17 | 68.95 | 90.74 |
| SOFC-Exp, random splits | 29.7 | 91.3 | 82.0 | 86.27 | 86.40 | 89.95 | 90.16 |
| SOFC-Exp, article folds | 46.4 | 89.3 | 79.7 | 82.79 | 82.94 | 88.54 | 88.81 |
| SOFC-Exp, official split | 49.3 | 88.7 | 82.0 | 83.75 | 84.06 | 88.48 | 88.80 |
| SOFC-Exp slot types, official split | 59.0 | 87.4 | 61.8 | 71.88 | 83.44 | 75.75 | 89.38 |
