Table 15. The lexicon split per unit (ner_lexicon_recall.json, post hoc): per setting and comparison, each held-out unit's recall gain on the unseen test names its lexicon holds and on the others, and their difference (the paired means are Table 10).

| setting | comparison | unit | gain, names in the lexicon | gain, names not in it | in − out |
|---|---|---|---|---|---|
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − MR6 | electrolysis | +6.40 | −0.61 | +7.01 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − MR6 | graphene | +0.48 | −1.57 | +2.05 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − MR6 | pemfc | +6.83 | +0.93 | +5.90 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − MR6 | poly | +4.64 | +2.03 | +2.61 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − MR6 | semi | +7.78 | +4.85 | +2.93 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − MR6 | sofc | +14.09 | +1.51 | +12.58 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − MR6 | steel | +10.78 | +0.68 | +10.10 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − CRF head | electrolysis | +10.80 | +3.04 | +7.76 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − CRF head | graphene | +1.18 | +2.20 | −1.02 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − CRF head | pemfc | +8.23 | +7.52 | +0.71 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − CRF head | poly | +8.45 | +5.19 | +3.26 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − CRF head | semi | +14.36 | +7.41 | +6.95 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − CRF head | sofc | +20.67 | +14.30 | +6.37 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − CRF head | steel | +11.01 | +4.71 | +6.30 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 − CRF head | electrolysis | +4.40 | +3.65 | +0.75 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 − CRF head | graphene | +0.70 | +3.77 | −3.07 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 − CRF head | pemfc | +1.40 | +6.59 | −5.19 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 − CRF head | poly | +3.81 | +3.16 | +0.65 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 − CRF head | semi | +6.58 | +2.56 | +4.02 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 − CRF head | sofc | +6.58 | +12.79 | −6.21 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 − CRF head | steel | +0.23 | +4.03 | −3.80 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR3 − CRF head | electrolysis | +3.02 | +3.81 | −0.79 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR3 − CRF head | graphene | −0.75 | +2.30 | −3.05 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR3 − CRF head | pemfc | +4.01 | +6.42 | −2.41 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR3 − CRF head | poly | +2.91 | +1.38 | +1.53 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR3 − CRF head | semi | +3.89 | +1.40 | +2.49 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR3 − CRF head | sofc | +8.31 | +11.43 | −3.12 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR3 − CRF head | steel | +3.52 | +2.82 | +0.70 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names (mixed corpus) − MR6 | electrolysis | +6.17 | −2.01 | +8.18 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names (mixed corpus) − MR6 | graphene | −0.81 | −1.25 | +0.44 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names (mixed corpus) − MR6 | pemfc | +7.21 | +1.39 | +5.82 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names (mixed corpus) − MR6 | poly | +4.53 | +2.62 | +1.91 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names (mixed corpus) − MR6 | semi | +3.43 | +0.86 | +2.57 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names (mixed corpus) − MR6 | sofc | +16.40 | +5.76 | +10.64 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names (mixed corpus) − MR6 | steel | +6.20 | +1.77 | +4.43 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 + names − MR6 | 10.1007_s11340-021-00758-x | +1.25 | −3.14 | +4.39 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 + names − MR6 | 10.1038_ncomms7164 | +13.48 | +0.61 | +12.87 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 + names − MR6 | 10.1038_s41467-020-19071-7 | −0.45 | −0.08 | −0.37 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 + names − MR6 | 10.3390_met10111458 | +4.19 | +3.78 | +0.41 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 + names − CRF head | 10.1007_s11340-021-00758-x | +0.00 | +6.86 | −6.86 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 + names − CRF head | 10.1038_ncomms7164 | +35.22 | +5.64 | +29.58 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 + names − CRF head | 10.1038_s41467-020-19071-7 | +2.96 | +3.42 | −0.46 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 + names − CRF head | 10.3390_met10111458 | +12.90 | +8.08 | +4.82 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 − CRF head | 10.1007_s11340-021-00758-x | −1.25 | +10.00 | −11.25 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 − CRF head | 10.1038_ncomms7164 | +21.74 | +5.03 | +16.71 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 − CRF head | 10.1038_s41467-020-19071-7 | +3.41 | +3.50 | −0.09 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 − CRF head | 10.3390_met10111458 | +8.71 | +4.30 | +4.41 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR3 − CRF head | 10.1007_s11340-021-00758-x | +0.00 | +8.00 | −8.00 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR3 − CRF head | 10.1038_ncomms7164 | +21.30 | +1.61 | +19.69 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR3 − CRF head | 10.1038_s41467-020-19071-7 | +4.09 | +3.24 | +0.85 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR3 − CRF head | 10.3390_met10111458 | +9.68 | +3.81 | +5.87 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 + names − MR6 | 10.1007_s11340-021-00758-x | +1.54 | +4.85 | −3.31 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 + names − MR6 | 10.1038_ncomms7164 | −0.41 | −0.16 | −0.25 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 + names − MR6 | 10.1038_s41467-020-19071-7 | +0.67 | +1.75 | −1.08 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 + names − MR6 | 10.3390_met10111458 | −0.34 | +1.20 | −1.54 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 + names − CRF head | 10.1007_s11340-021-00758-x | +1.15 | +5.84 | −4.69 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 + names − CRF head | 10.1038_ncomms7164 | +3.06 | +3.48 | −0.42 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 + names − CRF head | 10.1038_s41467-020-19071-7 | +0.67 | +2.55 | −1.88 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 + names − CRF head | 10.3390_met10111458 | +0.75 | +2.63 | −1.88 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 − CRF head | 10.1007_s11340-021-00758-x | −0.39 | +0.99 | −1.38 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 − CRF head | 10.1038_ncomms7164 | +3.47 | +3.64 | −0.17 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 − CRF head | 10.1038_s41467-020-19071-7 | +0.00 | +0.80 | −0.80 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 − CRF head | 10.3390_met10111458 | +1.09 | +1.43 | −0.34 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR3 − CRF head | 10.1007_s11340-021-00758-x | +0.38 | +2.97 | −2.59 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR3 − CRF head | 10.1038_ncomms7164 | +0.82 | +3.12 | −2.30 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR3 − CRF head | 10.1038_s41467-020-19071-7 | −1.26 | +1.19 | −2.45 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR3 − CRF head | 10.3390_met10111458 | −0.41 | +1.99 | −2.40 |
