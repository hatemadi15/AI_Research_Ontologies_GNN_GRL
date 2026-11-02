# Harvesting names from unlabeled papers to augment materials named entity recognition on unseen documents

Hatem Madi (affiliation to be added; hatem.madi15@gmail.com)

## Abstract

Materials named-entity recognizers trained on few annotated papers miss
names no training sentence annotates. In pre-registered tests we
added names three baseline taggers confidently agree on in 40,000
unlabeled sentences to mention replacement, which swaps names in copied
sentences. Over 11 held-out subfields and papers of two corpora this
gains 3.58 F1 points [95% CI 0.95, 6.21] over the baseline, three-copy
mention replacement alone 2.35 [0.94, 3.75] (post hoc). This pooled criterion
was adopted mid-study, when mention replacement met it; the
original, gains on both corpora, failed; a third corpus shows no gain.
Over six-copy mention replacement the names add 0.80 [−0.66, 2.26] on
seeds unused in selection, not established. On MuLMS they raise recall
on unseen names their lexicon holds, at lower precision (post hoc),
passing a registered test against a recall-matched decoding bias only
with three subfields it could not match.

## Introduction

Annotated materials corpora are small and come from few documents.
MaterioMiner<sup>1</sup> annotates 4 papers; SOFC-Exp<sup>2</sup> 45 articles; MuLMS<sup>3</sup> 50 articles
from 7 subfields. A named-entity tagger trained on such a corpus is meant
for documents the corpus does not contain, and a random sentence split
does not measure that. Our baseline is MatSciBERT<sup>4</sup>,
a BERT language model<sup>5</sup> pre-trained on materials text, with a conditional
random field (CRF) output layer: the *CRF head* (Methods, *Systems*). Scored by
micro F1 (the harmonic mean of precision and recall over entities matched
exactly in span and type, in percentage points; Methods, *Evaluation protocol*), it reaches
77.49 with MaterioMiner's fine-grained types (its 27 most frequent
classes; the coarse-grained level has 27 parent types, Table 1) over 5
random splits and 46.46 on a held-out paper; on MuLMS 65.72 over random
splits of the same size and 48.66 on a held-out subfield; on SOFC-Exp,
whose 45 articles share one topic and one annotation scheme, 86.27
against 82.79 on held-out articles. The gap is partly the names (an
entity's word string, values such as "10 mA cm−2" included): 81% of
the entities of a held-out MaterioMiner paper are *unseen*, their words
annotated as no entity in any training sentence (88% on a held-out MuLMS
subfield, 46% on SOFC-Exp's held-out articles), and the CRF head finds
35% of them on MaterioMiner and 39% on MuLMS, against 71% and 66% of the
names it has seen. The held-out document's mix of names accounts for a
third to three fifths of the recall gap, by corpus and by the order of
the decomposition; the rest is lower recall on both kinds (a companion
study, H.M., in preparation; Methods, *Relation to a companion study*).

Text mining has become a source of materials data<sup>6,7</sup>. Besides MatSciBERT,
ChemDataExtractor<sup>8</sup> combines dictionaries,
rules and a CRF tagger; Weston et al.<sup>9</sup> applied a
materials named-entity recognition (NER) model at scale; Trewartha et al.<sup>10</sup>
and MatSci-NLP<sup>11</sup> benchmark materials language models;
and large language models now extract structured materials information
by fine-tuning<sup>12</sup> or prompting<sup>13</sup>. Potu et al.<sup>14</sup> report 66.74 F1 on
MaterioMiner against 78.56 in distribution, a gap of about 12 points
where our three-paper models lose 28–31 points fine-grained and 17–20
coarse-grained (companion study). Random
sentence splits leak names between training and test<sup>15</sup>,
and their use is disputed<sup>16,17</sup>; taggers
score far lower on unseen mentions<sup>18–20</sup>, lean on name regularity and the
names themselves<sup>21,22</sup>, narrow but do
not close that gap with contextual embeddings<sup>23</sup>,
and are exposed when a test set's entities are switched<sup>24</sup>. Seed variance and significance on small
test sets are treated by Reimers and Gurevych<sup>25</sup>, Dror
et al.<sup>26</sup> and Bouthillier et al.<sup>27</sup>, and Card et al.<sup>28</sup> show how
underpowered typical natural language processing comparisons are, which is why Supplementary Note
9 reports what each comparison here could detect.

Mention replacement, which swaps annotated
mentions for others of the same type, is one of the simple augmentations
Dai and Adel<sup>29</sup> compared on a materials synthesis corpus<sup>30</sup> and a clinical one; it helped most with small
training sets, and its gains shrink and reverse as training grows<sup>31</sup>; most augmentations make the tagger memorize
entities<sup>32</sup>, which is why unseen entities are scored
separately here. The replacements otherwise come from the training set<sup>33</sup>, a masked language model<sup>34</sup>, a
generative model<sup>35</sup>, related sentences<sup>36</sup> or
lists of the target domain's names<sup>37</sup>; DAGA<sup>38</sup> and LADA<sup>39</sup> generate or interpolate labeled
examples, and cross-domain work<sup>40,41</sup>
changes domain, not document. We change only where the names come from,
keep every gold context, and test on held-out documents of the same
corpus.

Gazetteers have been
bootstrapped from unlabeled text since Riloff and Jones<sup>42</sup> and Kozareva<sup>43</sup>, and
agreement between models as a filter on unlabeled examples goes back to
co-training and tri-training<sup>44,45</sup>.
Zafarian and Asghari<sup>46</sup> keep a supervised tagger's
confident entities as a gazetteer feature; GEMNET<sup>47</sup> gates
gazetteer features against the context; Song et al.<sup>48</sup>
find no clear gain from replacing annotated entities with gazetteer
entries, on a corpus whose test names are mostly seen. AutoNER<sup>49</sup>, BOND<sup>50</sup> and weak supervision with
labeling functions<sup>51</sup> train taggers from dictionaries,
heuristics and unlabeled text; students learn from several teachers on
unlabeled target-language text<sup>52</sup>, and teachers' agreement
selects labels in distantly supervised NER<sup>53</sup>. Our
self-training variant follows BOND on the topic corpus, where Du et al.<sup>54</sup> retrieve sentences from the web. What this paper
adds is a pre-registered test of such a lexicon on held-out documents:
its contribution isolated against mention replacement with the same
number of copies, its recall compared with that of a constant added at
decoding to every entity tag's score, and its source and settings
varied, each decision fixed in a protocol committed before its
results.

This paper asks whether the unseen names can be given to the tagger without
annotation. Unlabeled materials text is
plentiful: a corpus of 40,000 sentences of open-access papers on a
held-out MuLMS subfield's topic holds 55–81% of its unseen test names.
Three CRF heads trained on the labeled sentences (the *teachers*) tag the
corpus; the names all three tag alike, each tag with a probability (its
*marginal*) of 0.9 or more, form a *lexicon*; and the lexicon enters
training through mention replacement<sup>29</sup>, which copies each
training sentence six times, each entity mention swapped with probability
0.5 for another mention of the same type, drawn from the training mentions and the
lexicon together (*MR6 + names*; Figure 1 and Methods, *Systems*). The model trained on the
augmented data, the *student*, never trains on a teacher-labeled
sentence: it sees only annotated sentences, some of whose names are
swapped for harvested ones. The Methods (*Terms used throughout*) collect these and the other terms.

Every decision was fixed in a protocol committed to the repository
before its results<sup>55</sup>, with 7 to 10
training runs (*seeds*) per held-out *unit* (a MuLMS subfield, a
MaterioMiner paper or a SOFC-Exp fold of articles) and a decision rule
stated in advance: a system improves on its reference when the 95%
interval of the mean paired difference in micro F1 over the units lies
above 0. A number in square brackets is that 95% Student-t confidence
interval, over the units named; a comparison specified after the results
of the runs it analyzes were known is labeled post hoc, and one fixed in
a protocol without a verdict rule of its own is labeled pre-specified,
with the results already known, if any, named. Protocols are
cited by date, with the time where one day has two, and Supplementary
Note 1 lists each with its commit. The *names route* is the sequence of
protocols that led to and tested harvested names; five of them (18
October 11:00 to 1 November) took the twelve *decisions* on MR6 + names,
D1 to D12 (Table 2), beside others counted at the end of the Results. The
results follow, with the qualifications recorded in the Methods
(*Evaluation protocol*).

MR6 + names improves on the CRF head on MaterioMiner's held-out papers
at both of its type levels (its *granularities*; D3, D4) and pooled over
the 11 units of MuLMS and MaterioMiner (D5; Results, *The pre-registered
decisions*), under a pooled criterion adopted mid-study, when plain
mention replacement was already known to meet it (Methods, *Evaluation
protocol*). MuLMS alone does not pass (D1).

Most of that gain is mention replacement's. Mention replacement with
three copies and no names (MR3) gains about two thirds of it on the same
units (post hoc), and the pre-registered test of the names against
six-copy mention replacement (MR6; D7) excludes an increment above 2.3
points but not a smaller one (Results, *Whose gain is it*). Macro F1
(every type weighted equally) and weighted F1 (types weighted by their
test entities) read the same way on the seeds not used in selection (the
*decision seeds*; post hoc: macro F1 +0.83 [−0.01, +1.67], P = 0.052,
weighted F1 +0.89 [−0.40, +2.18], P = 0.15; Results, *The pre-registered decisions*).

Neither the names' source nor the lexicon's settings changes F1
detectably (Results, *The names' source and the lexicon's settings*). On
MuLMS the names raise recall on unseen names, most on the fifth of them
the lexicon holds (+7.29 [+3.27, +11.30] over MR6, P = 0.004, post hoc), at lower
precision on those names (69.0% against 74.7%, pooled), so that F1 on
them is not shown to change (+1.13 [−1.85, +4.11], P = 0.39, post hoc). MR6 does
not reach the names' recall on them at any of twelve decoding biases
tried, constants from −2 to 5 added to every entity tag's score (post
hoc). The registered test against MR6 decoded at the bias that matches
the names' overall recall (D11) passes, but only with three subfields
that no bias matched (+6.05 [−3.33, +15.44], P = 0.13, on the four
matched), and the names are not shown to gain F1 over it (D12; Results, *What the
names do*). On SOFC-Exp, where half the test names are seen, no system
is shown to improve on the baseline (Results, *SOFC-Exp: no gain
shown*).

Two follow-ups were registered after D1 to D8 had been written up: the
recall-matched control (D11, D12) and a test of the names on
MaterioMiner's strongest system, multi-task training (one encoder with a
CRF head for each of MaterioMiner's two type levels) with mention
replacement, where against a six-copy matched control neither of its two
decisions passes (D9, D10; Results, *The names on the strongest
system*).

Four of the twelve decisions pass: three on F1, which share
MaterioMiner's four papers (D3 and D4 at its two granularities, and D5,
whose 11 units include them), and D11, on recall, which passes only
with three subfields that no bias matched and whose outcome was chosen
after a post hoc split of the unseen names by lexicon membership had
found the effect: it tests one alternative explanation, a uniform shift
toward tagging more spans, not the effect afresh. None would survive a
Bonferroni correction (Results, *The pre-registered decisions*), and
D3's interval would cover 0 if the training documents its units share
were allowed for (post hoc; Methods, *Statistics and reproducibility*).
The last part of the Results counts what else was tried on the way.

## Results

### The pre-registered decisions

Table 2 lists the twelve decisions of the names route in the order they
were taken, with the seeds each used; D9 and D10 belong to the follow-up of *The names on the strongest system*, D11 and D12 to that of *What the names do* (both below).

Three F1 decisions pass, and they are not three pieces of evidence: D3
and D4 are the same four papers at two granularities, and D5's eleven
units contain those four papers (the fourth pass, D11, is on recall and
holds only with three subfields that no bias matched; Results, *What the
names do*). On MuLMS, the corpus of the pooled decision that is independent of
those papers, MR6 + names fails D1 and D2, while MR6 alone clears 0
against the CRF head on the decision seeds (+2.68 [+0.13, +5.23], P =
0.042, pre-specified on 29 October; on all ten seeds +2.33 [−0.27,
+4.92], P = 0.071, covering 0, pre-specified when seeds 0–200 were
already known from the screen, the 3-seed comparison of variants that
selected MR6 + names), with
a smaller
mean than MR6 + names' +3.19 but a narrower spread; with MR6 + names 6 of
the 7 subfields gain (solid oxide fuel cells, sofc, by 11.86, while
polymers, poly, lose 2.35; Table 1 gives the subfields' short names), and the differences
have a standard deviation of 4.7 points across subfields. On
MaterioMiner every paper gains at both granularities, but with four
units the sign-flip test (the exact permutation test of the units'
signs; Methods, *Terms used throughout*) cannot go below P = 0.125, so D3 and D4 rest on
the t interval with 3 degrees of freedom. The pooled estimate survives
leaving out any one unit (post hoc: the lowest lower bounds are +0.44
and +0.54, without semiconductors), and the other post hoc checks agree:
weighting the two corpora equally gives 3.73 on either seed basis, with
Welch intervals [+1.39, +6.07] and [+1.46, +6.00] (D7's is +0.90
[−0.88, +2.69]); with MaterioMiner's coarse-grained papers in place of
its fine-grained ones the pooled estimate is +2.67 [+0.09, +5.25] (P =
0.044; Supplementary Note 12); 9 of the 11 units are positive on the executed 10-seed
basis and 10 on the 7-seed basis; and the exact sign-flip test gives
P = 0.011 and 0.007. Figure 2 shows every unit.

The decision rule carries no correction for the number of decisions.
Under a Bonferroni bound of 0.05/12 none of the four passes would hold:
their two-sided P values are 0.013 to 0.018 (0.011 for D5′, the post
hoc reading of D5 on MuLMS's decision seeds, not a decision; Table 2).
Mention replacement's post hoc pooled gain over the CRF head (Results,
*Whose gain is it*) has two-sided t-test P values of 0.0038 and 0.0040 on
the two seed bases, below 0.05/12 (0.0042), but as a post hoc comparison
it belongs to no corrected family.

**Other measures.** The protocols of 17 and 18 October named macro F1
(every type weighted equally), weighted F1 (types weighted by their test
entities) and the *votes* (the entity-level majority of the three models
of a seed triple) as reported beside micro F1; only mention
replacement's votes of 17 October decided. On MuLMS macro F1 varies far
less across subfields than micro F1, so its interval clears 0 in 23
per-corpus comparisons where micro F1's does not, every one on MuLMS and
against the CRF head or MR3, among them D1 (+3.78 [+2.47, +5.10]);
weighted F1's does so in 19 (16 on MuLMS, D1 among them at +4.17
[+0.39, +7.95], and 3 on MaterioMiner); the converse, micro F1 above 0
and macro F1 not, occurs 8 times, 7 on MaterioMiner or SOFC-Exp
(Supplementary Note 12). Against the names' own references on MuLMS's
decision seeds, and against MR6 pooled on the decision seeds, the
secondary measures give the micro picture. On MuLMS's decision seeds macro and
weighted F1 give +1.10 [−0.46, +2.65] and +1.17 [−1.09, +3.44] over MR3
(D2), +0.53 [−0.56, +1.62] and +0.84 [−1.04, +2.71] over MR6 (D7's MuLMS
part), and −0.24 [−1.56, +1.08] and −0.15 [−2.06, +1.75] against the
mixed corpus (D8). Pooled over D7's 11
units (post hoc), the names over MR6 give +0.83 [−0.01, +1.67] in macro
and +0.89 [−0.40, +2.18] in weighted F1 on the decision seeds, and +1.12
[+0.21, +2.03] and +1.22 [−0.21, +2.65] on all ten, which add the seeds
0 to 200 on which MR6 + names was selected. Over MR3 the pooled macro-F1
increment clears 0 (+1.33 [+0.29, +2.36] on the decision seeds, +1.73
[+0.84, +2.61] on all ten; weighted F1 +1.25 [−0.29, +2.78] and +1.53
[+0.26, +2.81]), the three extra copies included (MR6 over MR3, +0.50
[−0.31, +1.30] in macro and +0.36 [−0.11, +0.82] in weighted F1). In the
measures no decision used, the names' increment over the copy-matched
control thus has intervals that include 0 on the decision seeds (macro
F1's lower bound at −0.01, weighted F1's at −0.40), and one above 0 only
in macro F1 with the selection seeds. The votes read as micro F1 does
wherever they were computed: D1 +3.23 [−1.15, +7.61], D2 +0.57 [−2.22,
+3.36], D3 +4.99 [+3.27, +6.72], D4 +2.14 [+0.23, +4.05], D7's parts
+0.35 [−1.67, +2.36] (MuLMS) and +0.94 [−1.60, +3.49] (MaterioMiner), D8
−0.39 [−2.64, +1.86], D9 +0.19 [−2.28, +2.66] and D10 +0.63 [−1.65,
+2.91]; they were not computed for the pooled comparisons, for D11 and
D12, or on SOFC-Exp. Supplementary Note 12 lists all 162 rows of the names
route in every measure, with the votes and, where the results hold them,
the per-unit values.

### Whose gain is it

Table 3 gives every comparison among the four systems on the same 11
units, on both seed bases, and on MaterioMiner per granularity, with the
status of each.

The +3.58 decomposes, on the decision seeds, into +2.46 from mention
replacement with three copies, +0.33 from three more copies and +0.80
from the harvested names; only the first term has an interval above 0
(two-sided t-test P = 0.004, against 0.23 and 0.25 for the others), and
that post hoc (mention replacement's own pre-registered single-model
decisions passed on MaterioMiner and failed on MuLMS alone, where its
votes passed; Table 3 and Supplementary Note 7). The 10-seed basis gives
the same picture
(+2.35, +0.22, +1.02). Against MR3 the whole increment of MR6 + names,
+1.13, has an interval that covers 0 in micro F1 (in macro and weighted
F1, post hoc, +1.33 [+0.29, +2.36], P = 0.017, and +1.25 [−0.29,
+2.78], P = 0.10): the
pre-registered pass over the CRF head is
a property that mention replacement already has on these units (post
hoc). Three intervals in micro F1 comparing a names variant
with MR3 lie above 0, none of them a decision: MaterioMiner coarse-grained (Table
3, 3 degrees of freedom), the 2-of-3-teachers ablation on 3 seeds
(Supplementary Note 4) and SOFC-Exp (Results, *SOFC-Exp: no gain shown*). Per entity type the
pattern holds for most types (Supplementary Note 6).

D7's protocol foresaw a gain of +1.5 to +1.65 from the seeds in hand, an
11-unit half-width of about 2 and a chance of a pass of about 0.4; "not
established" is thus the likely outcome of a small effect under this
design, not evidence of none. The interval excludes a gain above +2.26,
nearly as large as mention replacement's whole gain (+2.46). The
follow-up on MaterioMiner's strongest system gives a similar increment
(Results, *The names on the strongest system*).

### The names' source and the lexicon's settings

**Source (D8).** For each held-out subfield a mixed corpus of 40,000
sentences, drawn equally from the six other subfields' corpora and built
the same way, holds 49–70% of the subfield's unseen test names against
the topic corpus's 55–81%. MR6 + names with the topic's own corpus scores
51.73 against 51.91 with the mixed one, −0.18 [−2.01, +1.65] paired by
subfield, in a comparison that detects 2.5 points. The mixed-corpus
variant gains +3.38 [−0.89, +7.64] (P = 0.10) over the CRF head, +1.10 [−0.76,
+2.97] over MR3 and +0.70 [−1.19, +2.58] over MR6 (pre-specified), its
own lexicon holding 18.3% of the unseen test names. The protocol fixed
this reading of a fail with the mixed variant above the CRF head: a
large materials corpus of other topics does as well, a simpler method.
The condition holds on the point estimate (+3.38) but not on the
interval, and "does as well" is an equivalence that a comparison
detecting 2.5 points cannot establish. An amendment of the same
protocol, made before any of D8's runs, had framed D8 more narrowly, as testing
whether the names the topic adds, and the contexts they come in, are
what the gain needs (the mixed corpora hold most of what the topic
corpora hold); D8 answers that they are not detectably.

**Settings (pre-specified, 3 seeds).** Five ablations, of corpus size
(10,000 and 20,000 sentences), threshold (0.7), agreement (2 of 3
teachers) and the names' share of the pool, keep micro F1 within 0.16 of
MR6 + names on the same seeds (those of the screen, the 3-seed comparison
of variants on which it was selected, a
slightly flattering reference), every interval against it within −2.45
and +2.14, while the lexicon ranges from 19 to 79 thousand distinct
names, summed over the seven subfields, and from 16% to 32% of the
unseen test names (Supplementary Note 4). Effects beyond those bounds are
excluded; smaller ones are not.

### What the names do

On MuLMS, pooled over the held-out subfields (post hoc, as is this
section up to the recall-matched control; Table 4; Figure 3 puts each system's recall and precision
on the unseen names on one plane with curves of equal F1), the CRF head
finds 39.4% of the unseen names at 55.8% precision, MR3 43.1% at 54.3,
MR6 43.7% at 53.9 and MR6 + names
47.3% at 50.9: against MR6, the copy-matched control, the names raise recall on
unseen names by 3.6 points and lower precision by 3.0, a net +0.7 F1 on
them, the point moving almost along a curve of equal F1. Paired by
subfield on the decision seeds, the recall gain is +2.56 [+0.76, +4.36],
the precision loss −2.22 [−5.03, +0.60] and the net
+0.56 [−1.44, +2.56]: only the recall gain's interval lies above 0 (P =
0.013, against 0.10 for the precision loss and 0.52 for the net; post hoc). On MaterioMiner's papers the pooled rates of both
recall and precision on unseen names rise over MR6, within paired
intervals that cover 0 (Supplementary Note 5).

**Which names the gain falls on (post hoc; the analysis and its reading
fixed before its run).** The lexicons were not stored with the runs; an
analysis of 31 October rebuilt each from its unit's own teachers and
corpus, reproducing the stored teachers' scores and coverage counts in
every unit, and split the unseen test names by whether the lexicon holds
them (Table 5; Supplementary Note 5 gives the reading rule and the full
analysis of 31 October (16:13)). On MuLMS the lexicon holds 22% of the
unseen test names,
pooled over the subfields (7% to 36% by subfield). On those, MR6 + names
recalls 82.4% against MR6's 75.0%; on the others, 36.8% against 35.0%;
the gain on the held names is the larger in every subfield, and the
difference of the two gains has an interval above 0 (Table 5; P = 0.007). The
reading rule fixed in advance (Supplementary Note 5) calls this *name
knowledge* if the other names' gain is read on its interval, which
covers 0, and *mixed* if on its point estimate (+1.12); MaterioMiner's
result is what the rule calls a *generic shift*, alike gains on both
parts, on point estimates, and none of its outcomes on intervals. The copies alone show no
detectable split (MR6 over the CRF head). Pooled precision falls on both
parts (74.7% to 69.0% on the held names, 46.4% to 44.0% on the others),
so that F1 on the held names is not shown to change (75.1 against 74.8
pooled; paired, +1.13 [−1.85, +4.11], P = 0.39; post hoc, Table 5), and two thirds
of the wrong predictions the names add, and 45% of the correct ones, are
names the lexicon does not hold. Because the teachers are the baseline, the
lexicon holds the unseen names the baseline already finds most easily:
the CRF head recalls 71.8% of them and 29.7% of the others (Table S3).

**Against a recall-matched control (follow-up; D11, D12).** A recall
gain paid for in precision might be no more than a shift toward tagging
more spans, which a constant bias at decoding would also give. The
follow-up of 1 November tested this on MuLMS at the decision seeds
(Supplementary Note 10). MR6 was run again and decoded with a bias D
added to the score of every tag but O (the tag of a word outside any
entity), over a grid from −2 to 5 (at D =
0 all 49 runs equal the stored ones in every score, in the training
history and in the predictions); per subfield the bias D* was found at
which MR6's recall over all test entities, pooled over the seeds, equals
that of MR6 + names, every quantity of the control at D* interpolated
linearly between the two grid values around it. At that bias the names
find more of the unseen test names their lexicon holds, +11.24 [+3.31,
+19.18], in every subfield (D11, passes as registered; exact sign-flip
P = 0.016, post hoc), and fewer of the other unseen names, −1.73 [−3.02, −0.44]
(P = 0.017;
pre-specified; −0.88 [−2.63, +0.87], P = 0.21, on the four matched subfields, a
reading fixed in the analysis code of 11:04, before any result); in micro F1 they are not shown to beat the control,
+3.26 [−0.79, +7.31] (D12), and in precision +5.78 [−1.22, +12.79]
(P = 0.090; pre-specified)
(Table 6, Figure 4). In three subfields, electrolysis, pemfc
(polymer-electrolyte-membrane fuel cells) and sofc,
no bias on the grid reaches the names' recall: MR6's recall peaks at D =
2.5 or 3, 0.2 to 0.5 points short, and falls beyond, where more spans
are tagged but fewer match exactly (Table S8). The protocol matched
these at the grid's end, D = 5, where MR6's recall is 1.7 to 2.9 points
below the names', and its seed-mean F1 5.3 to 7.0 points and its recall
on the held names 6.9 to 11.5 points below its own at D = 0, which
favors the names in both decisions. The protocol also reported the
decisions without these flagged subfields (pre-specified):
D11 +6.05 [−3.33, +15.44] (P = 0.13), all four subfields positive with 3 degrees of
freedom (an interval that could detect only about 12 points), and D12
−0.03 [−2.78, +2.72] (P = 0.97).

So +11.24 is the registered statistic, not the size of the effect. Per
subfield it is the names' gain over MR6 at D = 0 on the held names,
+7.29 on average (Table 5), plus MR6's own loss on them from D = 0 to
D*, +3.96 [−1.06, +8.98], nearly all of it in the three flagged
subfields (+0.14 on the other four; post hoc). Two post hoc readings
that do not depend on the protocol's matching of those three agree in
direction: with them matched at the grid value nearest the names'
recall, D11 is +8.63 [+3.35, +13.90] (P = 0.007) and D12 +1.37 [−0.66,
+3.41] (P = 0.15); and
against MR6 at whichever bias gives it its highest recall on the held
names, the names lead by +6.52 [+2.15, +10.89] (P = 0.011), in every subfield
(Figure 4), while MR6 at the bias best for the other unseen names
overtakes them there, −4.16 [−5.82, −2.50] (P = 0.0009; Supplementary
Note 10). No single decoding bias gives MR6 the names'
recall on the names the lexicon holds. A lexicon is mostly materials names and values
(83 to 99% of its distinct names; Supplementary Note 10), so a shift
toward those types could also favor the held names. Within types, at
D = 0, where no matching is involved, the names gain on the materials
names the lexicon holds and not detectably on the others (+7.98 [+2.15,
+13.82], P = 0.015, against +0.89 [−3.48, +5.27], P = 0.63; post hoc;
Table S9), and not detectably on the values it holds (+1.74 [−1.16,
+4.65], P = 0.19), though values
are most of every lexicon's distinct names; of the held names recalled
beyond MR6, about two thirds are materials names and a quarter are of
other types, whose held names are few but gain +9.80 [+3.34, +16.25]
(P = 0.010).
The gain is thus consistent with the student having learned names from
its lexicon, chiefly materials names (a post hoc reading, on MuLMS; which
of those names its copies contained was not measured). At the
protocol's matched bias it recalls fewer of the other unseen names,
though not detectably on the four matched subfields, and it is not shown
to gain F1 over the bias.

### SOFC-Exp: no gain shown

On SOFC-Exp's five folds of held-out articles MR6 + names with the SOFC
corpus scores 82.75 against the CRF head's 82.79, −0.04 [−1.21, +1.14]
(D6). It gains +0.45 [+0.04, +0.86] over MR3 on every fold
(pre-specified), but MR3 itself is not shown to gain there (−0.48 [−1.36,
+0.39], P = 0.20, a decision of 20 October (09:14); −0.29 [−1.04, +0.47],
P = 0.41, on the official split, paired by seed, a companion decision of 16 October (14:39), also failed). Two things differ on this
corpus and cannot be separated here: only 46% of a fold's test names are unseen (81%
fine- and 79% coarse-grained on MaterioMiner's papers, 88% on MuLMS's
subfields) and the CRF head already finds 80% of them; and the baseline
is at 82.8 F1 with a random-split gap of 3.5 points, so there is little
headroom. The SOFC corpus holds fewer of the unseen test names (36%,
27–43% per fold, measured before the runs and recorded in the protocol of
20 October, 13:53, whose measuring script was not committed) than MuLMS's topic corpora
(55–81%), but its lexicon holds 20–33% per fold, no less than MuLMS's or
MaterioMiner's, and against MR3 the names' interval lies above 0 there
(+0.45 [+0.04, +0.86], P = 0.038; pre-specified, one of many comparisons
without a decision): it is
mention replacement that is not shown to gain. Whether training size
explains that is open: trained on about as many entities as
MaterioMiner's training sets hold, MR3 gained +0.50 [−0.47,
+1.46] (P = 0.23), also not shown to differ from 0 (a decision of 17 October, 07:28;
Supplementary Note 7). Pooled over the 16 units of the three corpora the
estimate of MR6 + names is +2.45 [+0.50, +4.40] and that of MR3 +1.46
[+0.28, +2.64], and over the 9 units of MaterioMiner and SOFC-Exp that
of multi-task training with MR3 +2.93 [+0.14, +5.72] (specified on 20
October, MR3's and multi-task training's at 09:14, when the results of 11
of MR3's 16 units and of 4 of multi-task training's 9 were known, the
names' with D6 at 13:53, 18 minutes after the screen's seeds 0–200 of
its 7 MuLMS units were recorded); the claim stays the two-corpus one, SOFC-Exp a corpus
where no gain is shown.

### The names on the strongest system

The names had been tested only on the single-task CRF head. A follow-up
protocol, committed on 31 October after D1 to D8 had been written up and
before any of its runs, expecting a pass to be unlikely (Supplementary
Note 1), tested them on MaterioMiner's strongest system, multi-task
training with mention replacement: the names system against the
multi-task model with six-copy mention replacement, over 10 seeds on the
four held-out papers, fine-grained (D9) and coarse-grained (D10), its
lexicon harvested by three multi-task teachers with a filter over both
heads (Methods, *Systems*; Supplementary Note 8). Neither decision passes
(Table 2): fine-grained, all four papers are positive (0.16 to 1.62;
two-sided t-test P = 0.052, and a sign-flip P of 0.125, the smallest four
units allow); coarse-grained, three are. Over multi-task training with
three-copy mention replacement the names' whole increment is +0.76
[−0.15, +1.68] (P = 0.077) and +0.91 [−0.14, +1.97] (P = 0.070;
pre-specified; Table S6): on this system too what the names add is about a point and
not established. The fine-grained spread rests on 3 degrees of freedom,
so the interval, not the 1.3 points it would detect at that spread
(Table S7), is the result. The two-head filter's lexicons hold fewer of
the unseen test names than the single-task lexicons of D3 and D4 (by the
runs' stored counts, Supplementary Note 8: 7.5% against 9.6%
fine-grained, 12.1% against 18.3% coarse-grained); whether
that lowered the increment is not known.

### What else was tried, and the count

MR6 + names was reached through two earlier protocols and a screen of
seven variants on MuLMS's held-out subfields and their MaterioMiner
analogues, each committed before its runs, with two rules applied to
saved predictions (the lexicon as a dictionary, and label propagation
within a document); Supplementary Note 7 gives them in full, with the
count. In all, the protocols from 17 October (20:21) to 1 November took
twenty decisions on the names or on mention replacement: the twelve of
Table 2, the six of the two earlier protocols (four on MuLMS, two on
MaterioMiner) and two on SOFC-Exp's folds. Five passed: D3, D4, D5, D11
(as registered, only with three subfields that no bias matched) and
mention replacement's votes against the CRF head's votes. Of the
seven decisions of Table 2 against a mention-replacement control or on
the names' source, only D11, on recall, passes, as registered only with
three subfields that no bias matched, its outcome chosen after the
lexicon split had found the effect. If no method worked,
about 0.5 of twenty would be expected to pass, whatever their
dependence, which only makes chance passes come together, as the shared
units of D3 to D5 do. The count leaves out the label-propagation rule's
four decisions of 20 October, a rule on saved predictions rather than a
use of names: with them it is twenty-four, six passing, about 0.6
expected by chance (Supplementary Note 7); the companion study's
decisions, all earlier, are not counted.

## Discussion

A tagger trained on a few annotated materials papers misses more than
half of the entities of a new paper or subfield on MaterioMiner's
fine-grained types and on MuLMS. Mention replacement whose pool also
holds names mined from unlabeled text by three agreeing teachers
improves on the baseline on MaterioMiner's held-out papers at both
granularities (D3, D4) and pooled over the 11 held-out units of MuLMS
and MaterioMiner (+3.58 [+0.95, +6.21]; D5), though not on MuLMS alone
(D1). The pooled criterion was adopted mid-study, when mention
replacement without names was already known to meet it. Mention
replacement without the names gains most of that on the same units
(+2.35 [+0.94, +3.75], P = 0.004, post hoc), and the names' own increment in F1,
+0.80 [−0.66, +2.26] on the decision seeds (+0.83 [−0.01, +1.67], P =
0.052, in macro and +0.89 [−0.40, +2.18], P = 0.15, in weighted F1, post hoc), is not
established, nor on MaterioMiner's strongest system. What the names do
add, on MuLMS, is recall on the unseen names their lexicon holds, with
lower precision on them (69.0% against 74.7%), so that F1 on those names
is not shown to change (+1.13 [−1.85, +4.11], P = 0.39; post hoc). Post hoc, the
names model recalls more of them than MR6 does at any of the twelve
biases tried (+6.52 [+2.15, +10.89], P = 0.011). The registered recall-matched decision
(D11) passes, but only with three subfields where no bias reached the
names' overall recall (+6.05 [−3.33, +15.44], P = 0.13, on the four where one did),
and no F1 gain over the bias is shown (D12). This
is consistent with the student having learned names from its lexicon,
chiefly materials names; on MaterioMiner the reading rule fixed in
advance reads a generic shift, on point estimates.

For practitioners who fine-tune an encoder tagger on a few
hundred annotated sentences to read documents whose names are mostly
unseen in training (no large language model was compared), the advice
is mention replacement with three copies. Its gain on held-out documents
passed pre-registered on MaterioMiner (companion study; on the seven
seeds its screen did not use, post hoc, fine-grained +2.57 [+0.10,
+5.03] (P = 0.045) and coarse-grained +1.73 [−0.63, +4.08] (P = 0.10))
and is above 0
over the 16 units of the three corpora (+1.46 [+0.28, +2.64], P = 0.018,
specified on 20 October when 11 of the 16 units' results were known) and over the 11 of MuLMS and MaterioMiner (+2.35 [+0.94,
+3.75], P = 0.004, post hoc), while on MuLMS alone only its votes passed
(Supplementary Note 7). It costs about twice the baseline's training time
and needs no corpus. Neither three more copies nor the harvested names
are shown to add to it in micro F1 over the 11 units or in any decision,
here or on MaterioMiner's strongest system (three narrower comparisons
of a names variant with MR3 have intervals above 0; Results, *Whose gain
is it*); in the secondary measures, post hoc, the two together gain +1.33
[+0.29, +2.36] (P = 0.017) in macro and +1.25 [−0.29, +2.78] (P = 0.10)
in weighted F1 over it
on the decision seeds. A names student costs three to seven times an MR3
run, by how the harvest is charged, plus a 40,000-sentence corpus
(Supplementary Note 2); if names are used at all, a corpus of other
materials topics was not detectably worse than the held-out topic's own
(D8). The names shift recall toward the unseen names their lexicon
holds, by more than any decoding bias of the grid shifts MR6 (post hoc),
at a cost on the other unseen names against a recall-matched bias
(−1.73 [−3.02, −0.44], P = 0.017, pre-specified; −0.88 [−2.63, +0.87],
P = 0.21, on the four subfields it matched) though not against MR6 as
trained (+1.12 [−0.78, +3.02], P = 0.20, post hoc), and they are not shown to gain F1 (Results, *What the
names do*). Where recall on names that the baseline's own teachers tag
confidently matters more than precision, the names raise it (82.4%
against MR6's 75.0% on MuLMS), at three to seven times the cost of an
MR3 run and with 61% of their extra predictions of those names wrong;
the baseline itself already finds 72% of those names (Table S3), and
whether a dictionary rule over the same lexicon would give that recall
more cheaply was not measured: the probe of Supplementary Note 7, with
lexicons of CPU-trained teachers, raised recall over all test entities
in every subfield (by 1.4 to 11.2 points, at 2.8 to 24.2 points of
precision) but did not score the held names.
Even the best system finds only half of a held-out MuLMS
subfield's entities. If F1 is the goal, stop at
mention replacement, and expect no gain to be shown on a corpus like
SOFC-Exp, where half the test names are seen and the baseline is at 83
F1 (Results, *SOFC-Exp: no gain shown*).

Several limitations qualify these results. Seven subfields and four papers give 6
and 3 degrees of freedom; the subfields differ by 24 points under the
CRF head (32.49 to 56.82) and range from 2 to 11 articles, the papers
from 41 to 201 sentences, and the pooled decision weights them equally
as if drawn from one population. The leave-one-out and sign-flip
checks (Results, *The pre-registered decisions*) are post hoc, and with four units the sign-flip
test cannot go below P = 0.125, so D3 and D4 rest on the t interval
alone; a mixed model was not fitted, and with two corpora no
between-corpus variance could be estimated, so the pooled intervals are
conditional on these two corpora. Nor are the units independent:
MaterioMiner's papers and SOFC-Exp's folds share training documents, as
do 5 of the 21 pairs of MuLMS subfields; allowing for that (post hoc;
Methods, *Statistics and reproducibility*) leaves D4, D5 and D11 (as
registered, with its three unmatched subfields) above 0 and puts D3's
interval at [−0.13, +8.67]; D4's interval would reach 0 at a correlation
of 0.28 between papers, D5's at 0.45, and D11's at 0.53 between
subfields that share a training article. The pooled criterion's
condition that
neither corpus's own interval lie below 0 is weak: at the observed
spreads it would fail only if a corpus lost more than about 4.5 points
on average (MuLMS) or 2.9 (MaterioMiner).

The criterion under which MR6 + names passes
was adopted on 20 October on the author's decision to push for a
methods claim, after part of the screen and before the names system's
own screen result was recorded (its electrolysis runs had been pulled at
05:03, and whether their scores were read is not recorded), and when
mention replacement was already known to meet it (Methods, *Evaluation
protocol*); under the earlier criterion there is no claim:
D3, D4 and D6 would not have run, and D5 did not exist. The gate on D3 and D4 (that they run only if D1 passed) was last lifted at
18:42 that day, after the screen had chosen MR6 + names, by an
amendment whose reading from the screen's spread put D5 at about
[+0.1, +5.9]; the plan's estimate of a methods paper had fallen to
about 0.2 that morning. The screen
that chose MR6 + names and the MuLMS decisions share the same seven
subfields: the decision seeds keep the selection's seed noise out of
every MuLMS decision except D5 as executed, but not its fit to these
units.

Each MuLMS model
trains on a pool of 400 sentences (340 for training, 60 for
validation) of two or three of the other subfields'
articles (Table 1), a different set per held-out subfield with some
articles recurring across sets; the 88% share of unseen names and the
24-point spread between subfields reflect that.

Every positive result is at about 340 training
sentences; MuLMS's full pool was never used, SOFC-Exp's larger training
set is where no gain is shown, and the literature predicts smaller
gains for mention replacement as training grows. The claim is scoped to
a few hundred annotated sentences.

The names' effect is only partly isolated. The lexicon-membership
split (Results, *What the names do*) was run after D1 to D8, on rebuilt lexicons, and
its reading rule does not say whether a gain is read on its point
estimate or its interval. The recall-matched control (D11, D12) was
registered after D1 to D10 were known, with the names' side already
stored, and run on MuLMS only. Its grid could not reach the names'
recall in three of the seven subfields, which the protocol matched at
the grid's end, to the control's disadvantage; without them D11's
interval covers 0, and the readings that match them otherwise are post
hoc (Results, *What the names do*). It applies one bias to every tag, so it separates
name knowledge from a shift toward the lexicon's types only through the
post hoc split within types, and past its peak (D = 2.5 or 3) the bias
lengthens and merges spans; a bias on the tags that begin an entity was
not tried. The name-knowledge pattern is MuLMS's: on MaterioMiner the
split reads as a generic shift, and which lexicon names a student's
copies contained was not measured. The lexicon's
precision was not estimated (the precision of the predictions of
lexicon names, 69 to 75% on MuLMS and 81 to 83% on MaterioMiner, is the
nearest figure).

MR3's three copies and swap rate 0.5, MR6's six
copies, the 0.9 threshold and the three-teacher agreement were fixed in
advance, not tuned; Dai and Adel tune the copies (1 to 10) and the rate
(0.1 to 0.7) on development data, and a tuned mention replacement could
close or widen every gap in Table 3. The two-head filter of the follow-up of 31 October (19:15)
admits fewer names than the single-task one, and a looser one was not
tried.

Null results are not equivalences. The source decision and the
ablations detect differences of about 2 to 3 points (Supplementary Note
9); smaller effects of the source or the settings are not excluded.

The gazetteer-feature and self-training variants were
rejected on 3 seeds, and one encoder, MatSciBERT, was used. No large
language model was compared as an extractor, a teacher or a generator
of same-type names<sup>34,35</sup>; a generated pool is a
different source from names already present in unlabeled text and needs
its own matched test.

The corpora come from Europe PMC's
open-access subset, found by title terms, with journal concentration
and license mixes given in Supplementary Note 3, and are rebuilt from
hashed reports rather than redistributed. Pre-registration rests on the
repository's dated commits, whose times the author's machine sets, not
on an independent registry, though GitHub's own log of the pushes,
archived in the repository, puts every protocol commit of Table S1 (15
October to 1 November) on GitHub 1 to 7 seconds after its commit time
(Supplementary Note 1).

Several questions remain open. The first is a training-size curve on MuLMS, whose six other
subfields hold about 8,000 labeled sentences that the 340-sentence
regime leaves unused, since mention replacement's gains shrink with
training size<sup>29,31</sup>. Two variants
dismissed on MuLMS on 3 seeds deserve 10 there: self-training on the
corpus's teacher-labeled sentences, the one variant of the screen whose
interval against the CRF head cleared 0, though its MaterioMiner
analogue, the companion study's self-training on the fatigue corpus over
10 seeds, is not shown to beat MR3 (−1.03 [−4.47, +2.41], P = 0.41,
fine-grained, +0.28 [−1.46, +2.01], P = 0.65, coarse-grained, and its votes fall below MR3's
coarse-grained, −1.32 [−1.85, −0.80], P = 0.004; post hoc, both sides' results known when it was named); and
the lexicon as a gazetteer feature (Supplementary Note 7). A bias on the tags that begin an entity only, or
one per entity type, might match the three subfields that one bias on
every tag could not, since past its peak (D = 2.5 or 3) that bias
lengthens and merges spans (Supplementary Note 10); decoding MR6 + names
on the same grid would compare the two systems' whole recall–precision
curves, and a lexicon stripped of the test names it holds would show
whether the recall gain needs those very names. And the
lexicon's precision, never estimated, could be read from a sample of 100
names per type judged by a materials scientist; it would tell whether
the names fail to add F1 because they are wrong, redundant with the
training mentions, or already within the baseline's reach.

Every
decision, the criterion's revision and the amendments are reported. The
practical advice is mention replacement. The methodological advice is to
test a names method against a control with the same copies, and its
recall against a decoding bias checked beforehand to reach the method's
recall, on documents the training data has never seen, with enough
seeds and units to detect a point or two.

## Methods

### Terms used throughout

- *Unit.* A held-out MuLMS subfield, MaterioMiner paper or SOFC-Exp fold
  of articles, on which a system trained without it is scored.
- *Seed.* The random seed of a training run: 10 per unit (offsets 0 to
  900), or the 7 *decision seeds* (300 to 900) where a protocol used the
  other 3 for a *screen*, a 3-seed comparison that chooses among variants
  and decides nothing.
- *CRF head.* The baseline, MatSciBERT with a conditional random field
  output layer (Methods, *Systems*); *multi-task training* gives one encoder a CRF
  head for each of MaterioMiner's two type levels.
- *MR3, MR6.* Mention replacement with three or six copies of each
  training sentence, each mention swapped with probability 0.5 for one of
  the same type drawn from the *pool*: the training mentions, and for
  *MR6 + names* the lexicon too.
- *Teachers, marginal, lexicon, student.* Three CRF heads trained on the
  training sentences paired with a held-out unit (Table 1), never on the
  unit itself (seeds 0, 100 and 200), tag an unlabeled corpus; a tag's marginal is the probability a teacher gives it at its
  word; the lexicon is the mentions all three tag with the same span and
  type, every marginal at least 0.9; the student is the model trained on
  the gold sentences and their copies.
- *Decision, pre-specified, post hoc, follow-up.* A decision is a
  comparison whose verdict rule a protocol fixed before its results; a
  pre-specified comparison was fixed in a protocol without one, and one
  fixed when part of its results were known names them; a post
  hoc one was specified after the results of the runs it analyzes were
  known. D7, D8, D11 and D12 were registered when the system's results
  were known and before their references' new results; they count as
  decisions, with that disclosed (*Evaluation protocol*). The two *follow-ups* are
  the protocols of 31 October (19:15; D9,
  D10) and 1 November (10:51; D11, D12), registered after D1 to D8 had
  been written up. The *names route* is the protocols that led to and
  tested harvested names (17 October to 1 November).
- *Paired and pooled.* A paired difference is the mean over units of each
  unit's difference, each side the mean of its seeds (for a recall or
  precision, its rate over the seeds), with a 95% Student-t interval over
  the units. A pooled rate is a recall or precision from counts summed
  over seeds and units, without an interval; a pooled decision or estimate
  is one paired difference over the units of two or more corpora, with its
  interval.
- *Vote, sign-flip test.* A vote keeps an entity that at least two of
  three models predict with the same span and type; the three are the
  models of seeds 0–200, 300–500 or 600–800 (on the decision seeds the
  last two triples; seed 900 never votes), and a unit's vote score is the
  mean over its triples. The sign-flip test is the exact
  two-sided permutation test of a mean paired difference over all 2ⁿ
  assignments of signs to the n units.
- *Decoding bias, recall-matched control.* A constant D added at decoding
  to the score of every tag but O (the tag of a word outside any
  entity), so that a trained model tags more spans (or, below 0, fewer);
  the recall-matched control is MR6 decoded with the bias at which its
  recall over all test entities equals that of MR6 + names or, where no
  bias on the grid of twelve values from −2 to 5 reaches it (a *flagged*
  subfield), at the grid's end (Results, *What the names do*).
- *Seen and unseen names* are defined under *Data* below.

### Data

Table 1 gives the three corpora; the short names in parentheses are the
ones the figures and tables use. All three corpora are released under
open licenses; our conversions and the held-out protocol are in the
repository.

MaterioMiner's fine-grained level breaks ties among classes by token
count, then name (its 27th and 28th classes tie at 24 entities);
Supplementary Note 3 gives its papers' DOIs and MuLMS's kept entities by
type.

Three details matter for the Results. On MuLMS a flat tagger needs one level, so
31.4% of the entities are dropped with the conversion (14,632 of 46,584:
nested in a kept entity, sharing its span or crossing it), and text
extraction had damaged 276 tokens, which we repair; our MuLMS scores
cannot be compared with the published nested ones. On MuLMS the training
pool is capped at 400 sentences on purpose, so that MuLMS tests
MaterioMiner's regime, a few hundred annotated sentences from two or
three documents (SOFC-Exp, trained on 36 articles, does not; Results, *SOFC-Exp: no gain shown*); the other subfields hold about 8,000 labeled sentences that no run
here uses (Discussion). SOFC-Exp's official test articles are tokenized
differently from its training articles; the article folds used here cut
every article with the training articles' tokenizer.

**Unseen names.** A *name* is the word string of an entity of any type,
values such as "10 mA cm−2" included. A test entity is unseen when its
words are annotated as
no entity, of any type, in any training sentence, and seen when some
training sentence annotates them with its type<sup>18</sup>; entities whose words are annotated only
under another type, 0.0–2.1% of a corpus's test entities (up to 4.5% of
one paper's), count as neither. A predicted entity is classed by the same
rule from its words and its predicted type, so a correct prediction has
its gold entity's kind: recall on unseen names is the share of the
unseen gold entities found, and precision the share of the predictions
classed unseen that are correct. A name, gold or predicted, is in a
lexicon when the lexicon holds its words with its type.

### Evaluation protocol

**Seeds and scoring.** Every system compared in a decision has 10 seeds
per held-out unit (offsets 0, 100, …, 900), except where a protocol
reserved the screen's 3 seeds (0 to 200) for selection and decided on
the other 7; the mixed-corpus variant of D8 and the recall-matched
control of D11 and D12 have those 7 seeds, and the lexicon-setting
ablations the screen's 3 (Results, *The names' source and the lexicon's settings*). The score is micro F1 at the entity level: an
entity counts as found only with its exact span and type; precision is
the share of predicted entities that are correct, recall the share of
gold (annotated) entities found, F1 their harmonic mean, micro over all
entities of a test set (seqeval, github.com/chakki-works/seqeval, in its default
mode, as the datasets' own code scores), in percentage points.

**Decision rule.** A method improves on its reference when the 95%
Student-t interval of the mean paired micro-F1 difference lies above 0,
the pairs being the held-out units (7 subfields, 4 papers, 5 folds), each
side the mean of its seeds (D11 applies the rule to a recall; Results, *What the names do*); with 4 papers the interval has 3 degrees of freedom, with 7
subfields 6. A pooled decision takes the units of two
corpora together (11 units, 10 degrees of freedom) and requires, in
addition, that neither corpus's own interval lie below 0. Each decision
is one-sided at 2.5%, so of the twelve decisions about 0.3 would pass if
nothing worked (0.5 of the twenty counted at the end of the Results); the decisions
are not independent (Results, *The pre-registered decisions*). At 80% power and the observed spreads
the names against MR6
detect 2.0 points pooled over the 11 units and the names against the CRF
head 3.5 to 3.7 (Supplementary Note 9 gives every decision and the main
comparisons).

**Pre-registration, and what changed.** Each protocol was committed to
the repository before its results, with its decisions and its
pre-specified comparisons. All but three were committed before any run
of the system under test had started: D5's (as executed), that of D7 and
D8, and that of D11 and D12 (below). For D1 to D6 the references were
stored runs whose results were known; for D7, D8, D11 and D12 the
references were the new runs (MR6, the mixed-corpus variant, MR6
decoded under the bias) and the system's results were known; for D9 and
D10 both sides ran in the same sessions. Amendments are marked with their reason and time, and
Supplementary Note 11 lists every check, deviation and failure of the
names route. Supplementary Note 1 gives the commits and the
chronology, with GitHub's own log of the pushes, which puts every
protocol commit on GitHub 1 to 7 seconds after its commit time. The
protocols of the names route were fixed between 17 October and 1
November 2026. Three points must be disclosed.

First, the criterion for a methods claim was revised on 20 October at
09:14, on the author's decision, recorded in the protocol, to push for a
methods paper. Under the original criterion MR6 + names has no claim,
since D1 failed; D3, D4 and D6 would not have run, and D5 did not exist.
The changes, in order (UTC; Supplementary Notes 1 and 11):

- *18 October, 04:52.* The plan makes the MaterioMiner decisions of the
  protocol of 18 October (01:13) the gate of the route: topic corpora only if one passes, otherwise the
  evaluation paper.
- *18 October, 11:00.* The protocol of 11:00 replaces that gate with a
  coverage check, saying those decisions cannot run before 24 October.
  A methods claim needs a pass against the CRF head on MuLMS alone (D1)
  and one on MaterioMiner; the MaterioMiner analogue (the same system on
  MaterioMiner's papers, with a corpus on fatigue; D3, D4) runs only if
  D1 passes.
- *20 October, 00:44.* The MaterioMiner decisions of the protocol of 18
  October (01:13), run after all, are recorded: both fail, so under the plan of 04:52 the route would
  have stopped. The screen is pushed at 02:35.
- *05:03.* The screen's first session, which had run the names variants
  on electrolysis and MR6 on every subfield, is pulled; the names
  variants' outputs are not recorded, nor is it recorded whether their
  scores were read. They are identical to their reruns in every score,
  the training history and the predictions, and there MR6 + names gained
  less than MR6 (+2.06 against +2.97 over the CRF head). A batch with all
  the names variants' jobs is pushed before the quota is checked; the
  user is asked to cancel it, and the plan of 05:06 moves MR6 + names and
  the self-training variant to after 24 October, and the two MR3 + names
  variants to a smaller batch to be pushed in its place, which never was
  (Supplementary Note 11). The batch is not canceled; from 08:10 the
  plan expects Kaggle to stop it at the week's quota and to rerun it
  after 24 October.
- *08:11 and 08:13.* The screen's first part is recorded (MR6 and the two
  gazetteer-feature variants, none above MR3), and the plan's estimate of
  a methods paper falls from about 0.35–0.4 to about 0.2.
- *09:14.* The revision adds a pooled decision (D5) as a second route to
  the claim and lifts the gate on D3 and D4 and a stop rule (no fourth
  protocol on MuLMS). Mention replacement is known, from runs already made, to meet
  the pooled criterion, by a post hoc estimate: the protocol records
  +2.34 [+0.94, +3.75] with MaterioMiner's fine-grained papers (+2.35 in
  Table 3, from per-unit values rounded differently) and +1.80 [+0.39,
  +3.22] with the coarse-grained ones. The names of 18 October (01:13),
  from the other subfields' sentences on MuLMS and the fatigue corpus on
  MaterioMiner, meet it too (+2.39 [+0.71, +4.06], computed after the
  revision; post hoc; Supplementary Note 12). The names system's pooled result does not
  exist, and the screen has reported three of its seven variants, not
  MR6 + names.
- *13:35.* The screen's result is recorded, MR6 + names chosen; the batch
  of 05:03 had run on past the quota to its end at 12:25. Its +3.16 over
  the CRF head is read as clearing D1's bar of 2.4 to 2.9 points, and the
  plan puts D1 at about 0.5.
- *13:53.* An amendment re-imposes the gate on D3 and D4 and adds D6,
  SOFC-Exp's folds, on a pass of D1.
- *18:42.* An amendment lifts the gate on D3 and D4 again, on the ground that D5 needs
  the analogue whatever D1 gives, and widens D6's condition to a pass of
  D1 or D5. Its reading from the screen's own spread leaves D1 needing a
  mean gain near 5 and puts D5 at about [+0.1, +5.9]; the plan now puts
  D1 at about 0.3, D5 at about 0.4 and a methods paper at about 0.3.

D5's analysis was fixed before its data, but neither its adoption nor the
last lifting of the gate on D3 and D4, whose papers D5 needs, was blind
to related results. The conclusion
that the names' own increment is not established (D2, D7, D9, D10) rests
on neither.

Second, D5's protocol named MuLMS's decision seeds as the units' source
and, in the same paragraph, "each unit its 10-seed mean". The analysis as
executed (D5) averages in the screen's seeds 0 to 200, on which MR6 +
names was selected and whose runs were under way at 09:14, their results
not yet recorded; the decision-seed reading (D5′), which leaves them out,
agrees (Table 2), and the decomposition in the Results (*Whose gain is it*) uses it.

Third, the protocol of D7 and D8 (29 October) was committed two minutes
after its first batch was pushed and five hours before that batch's
first result, but after MR6 + names' own results on the same units (D1,
D3) were known: only MR6 and the mixed-corpus variant were new runs. That
of D11 and D12 (1 November, 10:51), registered at the request of an
internal review, run by the AI assistant, of the draft (*Use of an AI
assistant*, below) after D1 to D10 and the lexicon split (Results, *What the names do*) were known, is of
the same kind: the names' side is the stored runs, and only the control,
MR6 run again (identically) and decoded under the bias, was new. That of
D9 and D10 (31 October, 19:15) followed the write-up of D1 to D8 and
preceded its runs. Every other comparison, here or from the
earlier protocols, is labeled pre-specified or post hoc where it
appears.

**Runs.** Kaggle T4 GPUs with deterministic CUDA, one frozen commit per
batch, a pinned image and pinned libraries; model snapshots pinned by
commit. From 15 October 16:00 UTC every GPU session that completed, but
two, began by repeating an earlier run, and every repeat was identical
(the sessions that failed or hung left no record of their jobs); the
training settings are under *Systems*, and the compute and the repeats
in Supplementary Note 2.

### Systems

Figure 1 sketches MR6 + names, the last of the four systems compared
throughout:

- **CRF head.** MatSciBERT<sup>4</sup> with a token classifier on
  the first sub-word of each word and a linear-chain conditional random
  field (CRF) over the words, constrained to the IOB2 tag grammar (a tag
  that begins an entity, continues it, or is outside one), trained by
  likelihood and decoded with Viterbi, the highest-scoring tag sequence<sup>56,57</sup>, with early stopping on
  validation F1 (*Training*, below). It is the reference of D1 and D3
  to D6 and the
  baseline every system is measured from. It is not the strongest single
  model we know on every corpus: on MaterioMiner's held-out papers,
  fine-grained, multi-task training gains +3.72 [+0.57, +6.87] (P =
  0.033; pre-specified when 3 of its 10 seeds, 0–200, were already
  known) over it and multi-task training with three-copy mention
  replacement +6.73 [+6.03, +7.43] (P = 8 × 10⁻⁵; pre-specified; both
  without a decision in the companion evaluation study; Table S1); neither transfers
  to SOFC-Exp, and MuLMS has one type level, so the CRF head is the
  reference the three corpora share. The follow-up of 31 October (19:15) tested the names on
  that system, with six copies (Results, *The names on the strongest system*).
- **MR3.** Mention replacement<sup>29</sup>: three copies of each
  training sentence with an entity, each mention replaced with
  probability 0.5 by a training mention of the same type; copies that
  change nothing are dropped. The student trains on the gold sentences and
  the copies with the CRF head's training settings and seed.
- **MR6.** The same with six copies: the matched control for the names.
- **MR6 + names.** MR6 whose replacement pool also holds a lexicon
  harvested from unlabeled text:
  - *Teachers.* The CRF head's own runs of seeds 0, 100 and 200 on the
    unit's training sentences, so the teachers are the baseline, not a
    stronger model.
  - *Corpus.* 40,000 sentences of open-access full texts from Europe PMC<sup>58</sup> found by a query on titles for the held-out
    subfield (for MaterioMiner, "fatigue" in the title and a metal or
    process term, such as alloy, steel, titanium, weld or selective laser
    melting, in the title or abstract; for MuLMS's solid-oxide subfield,
    "solid oxide", "solid-oxide", "SOFC", "SOEC" and their plurals; for
    SOFC-Exp, "solid oxide", "SOFC" and "SOFCs"; every query is in the
    corpus reports), with the datasets' own articles excluded by
    identifier (for MuLMS, which ships no identifiers, the 16 of its 50
    articles found on Europe PMC; the rest are in journals Europe PMC does
    not index) and every sentence sharing a word 8-gram with a dataset
    sentence dropped (for MuLMS also a 50-character run). Sentences are
    split by a rule-based splitter that does not break after
    abbreviations or inside decimal numbers, and words by a regular
    expression that separates punctuation; sentences of 5 to 120 words
    with at least half alphabetic characters are kept, and formulas
    longer than 30 characters are cut. Each corpus has 290–368 articles
    with text (their composition in Supplementary Note 3). A topic corpus holds 55–81% of a held-out
    MuLMS subfield's unseen test names, the fatigue corpus 50–62% of a
    held-out paper's fine-grained (49–65% of its coarse-grained), the
    SOFC corpus 36% of the folds' (27–43% per fold; measured before the runs,
    Results, *SOFC-Exp: no gain shown*).
  - *Lexicon.* Each teacher tags the whole corpus and returns, per word,
    the probability the model assigns to the tag it chose (the
    forward–backward marginal of the Viterbi tag). A mention joins the
    lexicon when the three teachers tag the same span with the same type
    and every tag of the span has a marginal of at least 0.9 under each
    teacher, whatever the rest of the sentence. Per held-out MuLMS
    subfield this admits 8,328–50,869 mentions (4,380–13,395 distinct
    names), which hold 22.2% of the unseen test names pooled over the subfields
    (7–36% per subfield), about a third of what the corpus holds. The
    lexicon's precision was not estimated and its names were not stored
    with the runs (Discussion).
  - *Pool.* The training mentions (601–1,309 per MuLMS unit) and the
    harvested mentions, one entry per occurrence in the corpus, so that a
    draw is uniform over entries and a frequent name is drawn more often;
    harvested entries are at least 92% of a unit's pool. A mention is
    replaced only by an entry of its own type, so the lexicon's share of
    a mention's candidates depends on its type and subfield; the lexicons
    are chiefly materials names and values (83 to 99% of their distinct
    names; Table S10 gives the entries per type), and the share of
    replacements drawn from the lexicon was not measured.

**An example.** Take a training sentence, constructed for illustration,
"The Pt catalyst lowers the overpotential", with Pt annotated MAT and
overpotential PROPERTY. For the held-out electrolysis subfield, a copy
made by MR6 might read "The Ni catalyst lowers the overpotential", the
replacement drawn from the unit's training mentions, and a copy made by
MR6 + names "The MoS2 catalyst lowers the Tafel slope", both
replacements drawn from the subfield's lexicon. Its most frequent
entries are, as MAT, Ni, Pt, hydrogen, water, electrode, Co and catalyst,
and, as PROPERTY, current density and scan rate (each also in the
plural) and Tafel slope, and, as VALUE, 10 mA cm−2, 30 min, 2 h, 60 °C
and 1 M; values are 70% of its distinct names and 30% of its entries,
materials 23% and 64% (the rebuilt lexicon, which is not released; Results, *What the names do*; Supplementary Note 10); a
materials scientist might not count
electrode or catalyst as materials, and the lexicon's precision was not
estimated.

Every run stores its teachers' scores, the lexicon's counts and the share
of the unseen test names it holds, the corpus's hash and the filter. A
names student costs about 6 times the CRF head's run with the harvest
shared over its job's three or four students and about 14 times charged
to one student, not counting the corpus's construction (Supplementary
Note 2).

**Other systems.** The follow-up, the screen and the rules applied to
saved predictions use these (Results; Supplementary Notes 7 and 8):

- *Multi-task training* gives one encoder a CRF head for each of
  MaterioMiner's two type levels. The loss is the sum of the two heads'
  likelihood losses, each head keeps the weights of its own best
  validation epoch, and training stops when neither head has improved
  for 3 epochs. With mention replacement a coarse-grained mention is
  replaced, with probability 0.5, by one of the same coarse-grained type
  and the same fine-grained type (or none), in both tag sequences at
  once.
- *The two-head filter* (D9, D10) harvests with three multi-task
  teachers, the stored multi-task runs of seeds 0, 100 and 200, each
  tagging the fatigue corpus with both heads. A mention is admitted when
  the three agree on its coarse-grained span and type and on its
  fine-grained type at exactly that span (or on its having none), every
  marginal at least 0.9 on the coarse head and, where a fine type is
  tagged, on the fine head; it joins the pool under its pair of types.
- *The sentence filter* (the names of 18 October and two variants of the
  screen) keeps a whole corpus sentence when the three teachers tag it
  alike, with an entity and every marginal at least 0.9 under each
  teacher; the lexicon is then the mentions of those sentences.
- *Self-training* (a variant of the screen) trains the student on the
  gold sentences and those silver sentences, each silver sentence's
  likelihood weighted 0.5, the epoch chosen on the gold validation
  sentences.
- *The gazetteer feature* (two variants of the screen) adds to each
  word's emission scores a linear term, learned from zero (learning rate
  10⁻², no weight decay), on two indicators per entity type: whether the
  word begins, or continues, a verbatim match of a lexicon name of that
  type.
- *The dictionary rule* (a probe on saved predictions) tags every
  verbatim match of a lexicon name as its type where the CRF head
  predicted O.
- *The lexicon's settings* (five ablations, 3 seeds) harvest from 10,000
  or 20,000 of the corpus's 40,000 sentences (a fixed random sample, the
  first sentences of one seeded shuffle), at a marginal threshold of 0.7,
  or with two of the three teachers agreeing; or they draw a mention's
  replacement from the harvested names with probability 0.5 whenever its
  type has any, instead of uniformly over the pool's entries.
- *Label propagation* (a rule on saved predictions) tags the untagged
  occurrences, in a document, of a word string that the model tags in
  most of its occurrences there, with the type it is most often given.

**Training.** Every system trains MatSciBERT (110 million parameters)
with the CRF head's settings: AdamW<sup>59</sup> with a
learning rate of 5 × 10⁻⁵ and weight decay 0.01 for the encoder and
5 × 10⁻³ without weight decay for the CRF layer, a linear schedule that
warms up over the first 10% of the steps of 20 epochs and then decays
to 0, batch size 16, dropout 0.1, the gradient norm clipped at 1.0 and
every input within the encoder's 512 sub-words (none was longer, and
none was cut). Training stops when validation micro
F1 has not improved for 3 epochs (at most 20), and the best epoch's
weights are evaluated. The teachers' marginals come from the
forward–backward algorithm.

**The recall-matched control (D11, D12).** MR6's stored decision jobs on
MuLMS (400 pool sentences, six copies, seeds 300 to 900) were run again,
and each run was decoded at every bias D of the grid −2, −1, −0.5, 0,
0.5, 1, 1.5, 2, 2.5, 3, 4 and 5, D added to the emission score of every
tag but O before Viterbi decoding, training untouched. Per subfield,
recall over all test entities is pooled over the seven seeds (correct
entities summed over the seeds, divided by gold entities summed), for
MR6 + names and for the control at each D. The matched bias D* is the
first crossing of the names' recall by the control's, going from D = 0
toward it along the grid, found by linear interpolation between the two
grid values that bracket it, and every quantity of the control at D* is
the same interpolation of its grid values (pooled counts for the pooled
rates, each run's micro F1 for the seed mean). A subfield whose target
lies outside the control's range over the grid is matched at the nearer
end of the grid and flagged; it stays in the decisions, which are also
reported without the flagged subfields. D11 is recall on the unseen
test names the lexicon holds and D12 micro F1, each the names minus the
control at D*, paired over the seven subfields. D* is chosen on the test
set by design: the control tests a mechanism and is not a calibration one
could deploy (a bias chosen on validation folds, in the companion study,
gained nothing; Supplementary Note 5).

### Statistics and reproducibility

**Design.** The units are fixed by the corpora: MuLMS's 7 subfields,
MaterioMiner's 4 papers and SOFC-Exp's 5 folds of 9 articles. The
protocols fixed the units and the seeds (10 per unit, or the 7 decision
seeds) before the runs; Supplementary Note 9 gives the difference each
comparison could detect with 80% power at the observed spreads. No unit,
seed or run was excluded from a decision (the entities that the MuLMS
conversion drops are described under *Data*). A run's seed sets its
random choices (initialization, dropout, batch order and the swaps of
mention replacement), and a fixed split seed sets the training pools,
validation sentences and folds. There was no blinding: the same scripts
score every system, and the protocols, committed before their results,
stand in for it.

**Tests.** The decision rule (*Evaluation protocol*) is a one-sample
Student's t-test of the mean paired difference against 0: its interval
lies above 0 exactly when the one-sided P value is below 0.025, that is,
when the two-sided P value is below 0.05 with a positive mean; a pooled
decision also requires that neither corpus's own interval lie below 0.
A test's n is its number of held-out units (4 papers, 5 folds, 7
subfields or 11 pooled units; Table 2), each unit's value the difference
of the two systems' means over the unit's 10 seeds (7 on MuLMS's
decision seeds), so seeds are not counted as independent observations;
for D11 it is the difference of two recalls pooled over the seeds, the
control's interpolated at a bias chosen on the test set (*The
recall-matched control*). Table 2 gives each decision's two-sided P
value, and Table S16 its t statistic, degrees of freedom (n − 1;
Supplementary Note 13), one- and two-sided P values and the standard
deviation of its per-unit differences, with the exact sign-flip P value
beside them. They were computed on 1 November for the journal's
reporting rules, from the per-unit differences the decisions used, and
change no decision.

**Assumptions.** The t-test assumes roughly normal per-unit differences.
With 4 to 11 units a test of normality has little power, and none was
run. The exact two-sided sign-flip test, which assumes only that each
difference is symmetric about 0 under the null hypothesis, is given
beside it (post hoc): it agrees for D5 (P = 0.011), for its
decision-seed reading D5′ (0.007) and for D11 (0.016, as registered), but with 4 units
it cannot go below 0.125, so D3 and D4 rest on the t-test's assumption.
Both tests also treat the units' differences as independent. They are
not fully so: the training sets of any two MaterioMiner papers share two
of their three papers, those of any two SOFC-Exp folds 27 of their 36
articles, and 5 of the 21 pairs of MuLMS subfields share an article of
their training pools, and positively correlated differences make the
intervals too narrow. As a sensitivity analysis (post hoc), every two
units whose training sets share documents were given the correlation
that the corrected resampled t-test assumes<sup>60</sup>, the
held-out share of the data, 1/K in a design of K folds: 1/4 for MaterioMiner's papers and,
conservatively, for the MuLMS pairs, 1/5 for SOFC-Exp's folds, and none
across corpora; with S² the sample variance of the n per-unit
differences, the variance of their mean is then S²/(1 − ρ̄)·(n + Σρ)/n²,
Σρ over the n(n − 1) ordered pairs of units and ρ̄ = Σρ/(n(n − 1)), with
n − 1 degrees of freedom (at ρ = 1/K for every pair this is the corrected
resampled test's variance). D3 then reads +4.27 [−0.13, +8.67] (P = 0.054), D4
+1.77 [+0.09, +3.45] (P = 0.044), D5 +3.58 [+0.27, +6.88] (P = 0.037;
D5′, the decision-seed reading, [+0.38, +6.78], P = 0.032) and D11, as
registered with its three unmatched subfields, +11.24 [+1.71, +20.77]
(P = 0.028), and no other
decision's interval clears 0 (Table S16); the
decisions stand as registered. The passes turn on the assumed
correlation: D4's interval reaches 0 at a correlation of 0.28 between
MaterioMiner's papers, D5's at 0.45 (the MuLMS pairs held at 1/4) and
D11's at 0.53 between MuLMS subfields that share a training article
(`post_hoc_dependence` of `ner_decision_tests.json`). Figure 2 and Supplementary Note 12 show
the per-unit values; the Discussion gives what the pooled intervals
assume about the corpora.

**Multiplicity.** No level was adjusted for the number of decisions: the
protocols registered each rule unadjusted, and the count is reported
instead (Results, *What else was tried, and the count*). Under a
Bonferroni bound of 0.05/12 none of the four passes would hold (Results,
*The pre-registered decisions*).

**Estimates.** Every other interval in the paper is the 95% Student-t
interval of a paired difference over held-out units (*Terms used
throughout*), except where marked: the Welch intervals of the
corpus-weighted estimates (Results, *The pre-registered decisions*), the
comparisons paired by seed on SOFC-Exp's official split and of the
companion study's recall calibration (Results, *SOFC-Exp: no gain
shown*; Supplementary Note 5), the training-size comparison over five
draws of training articles (Supplementary Note 7), the intervals of a
system's level rather than of a difference (Table S6; Table S11, row
66) and those of a standard deviation (Supplementary Note 9). Such
intervals accompany pre-specified and post hoc comparisons as estimates, not
tests; "clears 0" says only that the lower bound is above 0. Each interval of a paired difference over units excludes 0 exactly when the two-sided P value of
the one-sample t-test behind it is below 0.05; where the text draws an
inference from one, it gives that P value (computed on 1 and 2 November for
reporting, `post_hoc_reported` of `ner_decision_tests.json`). Pooled
rates carry no interval. The difference a comparison could detect with
80% power (Supplementary Note 9) is SE·(t₀.₉₇₅,ν + t₀.₈₀,ν), SE being
its interval's half-width divided by t₀.₉₇₅,ν and ν its degrees of
freedom.

**Reproducibility.** Every run is deterministic, and every repeat of a
run was identical (*Evaluation protocol*; Supplementary Note 2). The
repository holds the code, every results file with its seeds and saved
predictions, and the scripts behind every number (Code availability).

### Use of an AI assistant

An AI assistant was used, under the
author's direction, to draft the protocols, write the code, run the
experiments, draft the text of every section from the results files and
run adversarial internal reviews of the drafts, among them the review
that asked for the recall-matched control (D11, D12). Every result in
the paper was taken from the outputs of the repository's analysis
scripts, and every date, odds figure and protocol-quoted number from its
dated records, and checked against them by those reviews, which the same
assistant ran. The author took the study's decisions, among them the
revision of the criterion on 20 October and the approval of the
recall-matched control's runs on 1 November, and is responsible for the
content.

### Relation to a companion study

This paper shares
data, baselines and protocol with a companion evaluation study of the
same program (H.M., in preparation; its draft is in the repository), which
reports the random-split gap, seed variance and the other additions
tested. The two papers overlap in the Methods. The results that
appear in both are the random-split gap and the unseen-name shares of
the Introduction, mention replacement's MaterioMiner decisions and the multi-task results in the Methods (*Systems*), mention replacement on SOFC-Exp's official split and at MaterioMiner's training size (Results, *SOFC-Exp: no gain shown*), the recall calibration of Supplementary Note 5, the label-propagation rule of
Supplementary Note 7, which the companion reports in its survey, and
self-training on the fatigue corpus over MaterioMiner's papers (the
companion's decision of 15 October), whose comparison with MR3 the Discussion
quotes; the
companion summarizes D1–D8 and the decomposition in the Results (*Whose gain is it*) in one table and cites this paper for them.

## Data availability

The labeled corpora are public: MaterioMiner, release v1.0.1 (Fordatis,
https://doi.org/10.24406/fordatis/329.2; CC BY 4.0), whose files the
repository includes; SOFC-Exp (github.com/boschresearch/sofc-exp_textmining_resources;
CC BY 4.0; at commit bbf75fd), whose converted files the repository
includes; and MuLMS (Hugging Face,
huggingface.co/datasets/timo-pierre-schrader/MuLMS at revision bc1c97d;
CC BY-SA 4.0). The repository, github.com/hatemadi15/AI_Research_Ontologies_GNN_GRL,
holds every results file with its seeds, environment and saved test
predictions, and the corpus reports with their queries, article
identifiers, licenses and SHA-256 hashes. No personal data is involved.
The unlabeled corpora, built from Europe PMC's open-access full texts,
are not redistributed, 10–33% of their articles being under non-commercial or
no-derivative licenses; they rebuild from their reports, with hash
checks, and the reports identify the CC BY articles. The runs did not
store the harvested lexicons; `ner_baseline.py --students 0 --save-names`
rebuilds each from its corpus (the lexicon-split analysis of 31
October), and they are not released, because they derive from corpora
that are not redistributed.

## Code availability

The repository, github.com/hatemadi15/AI_Research_Ontologies_GNN_GRL
(Apache 2.0; the commit this paper describes will be tagged, and an
archived copy with a DOI deposited, at submission), holds the code, every
protocol with its dates and amendments, the analyses behind every number,
the generated tables and figures, and the Kaggle batch definitions that
produced each results file.

## Acknowledgements

The author thanks the creators of MaterioMiner, SOFC-Exp and MuLMS for releasing their corpora under open licenses. This study received no funding; the GPU compute was Kaggle's free tier.

## Author contributions

H.M. conceived and directed the study, took its decisions and is responsible for its content.

## Competing interests

The author declares no financial or non-financial competing interests.

## References

1. Durmaz, A. R., Thomas, A., Mishra, L., Murthy, R. N. & Straub, T. An ontology-based text mining dataset for extraction of process-structure-property entities. *Sci. Data* **11**, 1112 (2024). https://doi.org/10.1038/s41597-024-03926-5
2. Friedrich, A. et al. The SOFC-Exp Corpus and Neural Approaches to Information Extraction in the Materials Science Domain. In *Proc. 58th Annual Meeting of the Association for Computational Linguistics* 1255–1268 (Association for Computational Linguistics, 2020). https://doi.org/10.18653/v1/2020.acl-main.116
3. Schrader, T. P., Finco, M., Grünewald, S., Hildebrand, F. & Friedrich, A. MuLMS: A Multi-Layer Annotated Text Corpus for Information Extraction in the Materials Science Domain. In *Proc. Second Workshop on Information Extraction from Scientific Publications* 84–100 (Association for Computational Linguistics, 2023). https://doi.org/10.18653/v1/2023.wiesp-1.11
4. Gupta, T., Zaki, M., Krishnan, N. M. A. & Mausam. MatSciBERT: A materials domain language model for text mining and information extraction. *npj Comput. Mater.* **8**, 102 (2022). https://doi.org/10.1038/s41524-022-00784-w
5. Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In *Proc. 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)* 4171–4186 (Association for Computational Linguistics, 2019). https://doi.org/10.18653/v1/N19-1423
6. Olivetti, E. A. et al. Data-driven materials research enabled by natural language processing and information extraction. *Appl. Phys. Rev.* **7**, 041317 (2020). https://doi.org/10.1063/5.0021106
7. Kononova, O. et al. Opportunities and challenges of text mining in materials research. *iScience* **24**, 102155 (2021). https://doi.org/10.1016/j.isci.2021.102155
8. Swain, M. C. & Cole, J. M. ChemDataExtractor: A Toolkit for Automated Extraction of Chemical Information from the Scientific Literature. *J. Chem. Inf. Model.* **56**, 1894–1904 (2016). https://doi.org/10.1021/acs.jcim.6b00207
9. Weston, L. et al. Named Entity Recognition and Normalization Applied to Large-Scale Information Extraction from the Materials Science Literature. *J. Chem. Inf. Model.* **59**, 3692–3702 (2019). https://doi.org/10.1021/acs.jcim.9b00470
10. Trewartha, A. et al. Quantifying the advantage of domain-specific pre-training on named entity recognition tasks in materials science. *Patterns* **3**, 100488 (2022). https://doi.org/10.1016/j.patter.2022.100488
11. Song, Y., Miret, S. & Liu, B. MatSci-NLP: Evaluating Scientific Language Models on Materials Science Language Tasks Using Text-to-Schema Modeling. In *Proc. 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* 3621–3639 (Association for Computational Linguistics, 2023). https://doi.org/10.18653/v1/2023.acl-long.201
12. Dagdelen, J. et al. Structured information extraction from scientific text with large language models. *Nat. Commun.* **15**, 1418 (2024). https://doi.org/10.1038/s41467-024-45563-x
13. Polak, M. P. & Morgan, D. Extracting accurate materials data from research papers with conversational language models and prompt engineering. *Nat. Commun.* **15**, 1569 (2024). https://doi.org/10.1038/s41467-024-45914-8
14. Potu, S. T. et al. Ontology-conformal recognition of materials entities using language models. *Sci. Rep.* **15**, 18597 (2025). https://doi.org/10.1038/s41598-025-03619-y
15. Babl, F., Hennen, M., Murauer, J. & Geierhos, M. Random Splitting Negatively Impacts NER Evaluation: Quantifying and Eliminating the Overestimation of NER Performance. In *Findings of the Association for Computational Linguistics: ACL 2025* 9724–9738 (Association for Computational Linguistics, 2025). https://doi.org/10.18653/v1/2025.findings-acl.504
16. Gorman, K. & Bedrick, S. We Need to Talk about Standard Splits. In *Proc. 57th Annual Meeting of the Association for Computational Linguistics* 2786–2791 (Association for Computational Linguistics, 2019). https://doi.org/10.18653/v1/P19-1267
17. Søgaard, A., Ebert, S., Bastings, J. & Filippova, K. We Need To Talk About Random Splits. In *Proc. 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume* 1823–1832 (Association for Computational Linguistics, 2021). https://doi.org/10.18653/v1/2021.eacl-main.156
18. Augenstein, I., Derczynski, L. & Bontcheva, K. Generalisation in named entity recognition: A quantitative analysis. *Comput. Speech Lang.* **44**, 61–83 (2017). https://doi.org/10.1016/j.csl.2017.01.012
19. Fu, J., Liu, P. & Zhang, Q. Rethinking Generalization of Neural Models: A Named Entity Recognition Case Study. *Proc. AAAI Conf. Artif. Intell.* **34**, 7732–7739 (2020). https://doi.org/10.1609/aaai.v34i05.6276
20. Kim, H. & Kang, J. How Do Your Biomedical Named Entity Recognition Models Generalize to Novel Entities? *IEEE Access* **10**, 31513–31523 (2022). https://doi.org/10.1109/ACCESS.2022.3157854
21. Lin, H. et al. A Rigorous Study on Named Entity Recognition: Can Fine-tuning Pretrained Model Lead to the Promised Land? In *Proc. 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)* 7291–7300 (Association for Computational Linguistics, 2020). https://doi.org/10.18653/v1/2020.emnlp-main.592
22. Ghaddar, A., Langlais, P., Rashid, A. & Rezagholizadeh, M. Context-aware Adversarial Training for Name Regularity Bias in Named Entity Recognition. *Trans. Assoc. Comput. Linguist.* **9**, 586–604 (2021). https://doi.org/10.1162/tacl_a_00386
23. Taillé, B., Guigue, V. & Gallinari, P. Contextualized Embeddings in Named-Entity Recognition: An Empirical Study on Generalization. In *Advances in Information Retrieval: 42nd European Conference on IR Research (ECIR 2020), Part II* 383–391 (Springer, 2020). https://doi.org/10.1007/978-3-030-45442-5_48
24. Agarwal, O., Yang, Y., Wallace, B. C. & Nenkova, A. Entity-Switched Datasets: An Approach to Auditing the In-Domain Robustness of Named Entity Recognition Models. Preprint at https://arxiv.org/abs/2004.04123 (2020).
25. Reimers, N. & Gurevych, I. Reporting Score Distributions Makes a Difference: Performance Study of LSTM-networks for Sequence Tagging. In *Proc. 2017 Conference on Empirical Methods in Natural Language Processing* 338–348 (Association for Computational Linguistics, 2017). https://doi.org/10.18653/v1/D17-1035
26. Dror, R., Baumer, G., Shlomov, S. & Reichart, R. The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing. In *Proc. 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* 1383–1392 (Association for Computational Linguistics, 2018). https://doi.org/10.18653/v1/P18-1128
27. Bouthillier, X. et al. Accounting for Variance in Machine Learning Benchmarks. In *Proc. Machine Learning and Systems* 747–769 (2021). https://arxiv.org/abs/2103.03098
28. Card, D. et al. With Little Power Comes Great Responsibility. In *Proc. 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)* 9263–9274 (Association for Computational Linguistics, 2020). https://doi.org/10.18653/v1/2020.emnlp-main.745
29. Dai, X. & Adel, H. An Analysis of Simple Data Augmentation for Named Entity Recognition. In *Proc. 28th International Conference on Computational Linguistics* 3861–3867 (International Committee on Computational Linguistics, 2020). https://doi.org/10.18653/v1/2020.coling-main.343
30. Mysore, S. et al. The Materials Science Procedural Text Corpus: Annotating Materials Synthesis Procedures with Shallow Semantic Structures. In *Proc. 13th Linguistic Annotation Workshop* 56–64 (Association for Computational Linguistics, 2019). https://doi.org/10.18653/v1/W19-4007
31. Elwing Torres, A., Silva de Moura, E., Soares da Silva, A., Nascimento, M. A. & Mesquita, F. An experimental study on data augmentation techniques for named entity recognition on low-resource domains. *Nat. Lang. Process.* **32**, 184–203 (2026). https://doi.org/10.1017/nlp.2026.10019
32. Sharma, S. et al. When and how to paraphrase for named entity recognition? In *Proc. 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* 7052–7087 (Association for Computational Linguistics, 2023). https://doi.org/10.18653/v1/2023.acl-long.390
33. Bartolini, I. et al. Named Entity Recognition using context similarity data augmentation. In *Proc. 32nd Symposium on Advanced Database Systems (SEBD 2024), CEUR Workshop Proceedings Vol. 3741* 331–338 (2024).
34. Zhou, R. et al. MELM: Data Augmentation with Masked Entity Language Modeling for Low-Resource NER. In *Proc. 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* 2251–2262 (Association for Computational Linguistics, 2022). https://doi.org/10.18653/v1/2022.acl-long.160
35. Hu, X. et al. Entity-to-Text based Data Augmentation for various Named Entity Recognition Tasks. In *Findings of the Association for Computational Linguistics: ACL 2023* 9072–9087 (Association for Computational Linguistics, 2023). https://doi.org/10.18653/v1/2023.findings-acl.578
36. Cai, J. et al. Graph Propagation based Data Augmentation for Named Entity Recognition. In *Proc. 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)* 110–118 (Association for Computational Linguistics, 2023). https://doi.org/10.18653/v1/2023.acl-short.11
37. Amalvy, A., Labatut, V. & Dufour, R. Remplacement de mentions pour l'adaptation d'un corpus de reconnaissance d'entités nommées à un domaine cible. In *Actes de la 29e Conférence sur le Traitement Automatique des Langues Naturelles. Volume 1 : conférence principale* 198–205 (2022).
38. Ding, B. et al. DAGA: Data Augmentation with a Generation Approach for Low-resource Tagging Tasks. In *Proc. 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)* 6045–6057 (Association for Computational Linguistics, 2020). https://doi.org/10.18653/v1/2020.emnlp-main.488
39. Chen, J., Wang, Z., Tian, R., Yang, Z. & Yang, D. Local Additivity Based Data Augmentation for Semi-supervised NER. In *Proc. 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)* 1241–1251 (Association for Computational Linguistics, 2020). https://doi.org/10.18653/v1/2020.emnlp-main.95
40. Chen, S., Aguilar, G., Neves, L. & Solorio, T. Data Augmentation for Cross-Domain Named Entity Recognition. In *Proc. 2021 Conference on Empirical Methods in Natural Language Processing* 5346–5356 (Association for Computational Linguistics, 2021). https://doi.org/10.18653/v1/2021.emnlp-main.434
41. Liu, Z. et al. CrossNER: Evaluating Cross-Domain Named Entity Recognition. *Proc. AAAI Conf. Artif. Intell.* **35**, 13452–13460 (2021). https://doi.org/10.1609/aaai.v35i15.17587
42. Riloff, E. & Jones, R. Learning Dictionaries for Information Extraction by Multi-Level Bootstrapping. In *Proc. Sixteenth National Conference on Artificial Intelligence (AAAI-99)* 474–479 (AAAI Press, 1999).
43. Kozareva, Z. Bootstrapping Named Entity Recognition with Automatically Generated Gazetteer Lists. In *Proc. Eleventh Conference of the European Chapter of the Association for Computational Linguistics: Student Research Workshop* 15–21 (Association for Computational Linguistics, 2006).
44. Blum, A. & Mitchell, T. Combining labeled and unlabeled data with co-training. In *Proc. Eleventh Annual Conference on Computational Learning Theory* 92–100 (1998). https://doi.org/10.1145/279943.279962
45. Zhou, Z.-H. & Li, M. Tri-training: exploiting unlabeled data using three classifiers. *IEEE Trans. Knowl. Data Eng.* **17**, 1529–1541 (2005). https://doi.org/10.1109/TKDE.2005.186
46. Zafarian, A. & Asghari, H. Improving NER Models by Exploiting Named Entity Gazetteer as External Knowledge. In *Proc. First International Workshop on NLP Solutions for Under Resourced Languages (NSURL 2019) co-located with ICNLSP 2019 – Short Papers* 107–111 (Association for Computational Linguistics, 2019).
47. Meng, T., Fang, A., Rokhlenko, O. & Malmasi, S. GEMNET: Effective Gated Gazetteer Representations for Recognizing Complex Entities in Low-context Input. In *Proc. 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* 1499–1512 (Association for Computational Linguistics, 2021). https://doi.org/10.18653/v1/2021.naacl-main.118
48. Song, C. H., Lawrie, D., Finin, T. & Mayfield, J. Improving Neural Named Entity Recognition with Gazetteers. In *Proc. Thirty-Third International Florida Artificial Intelligence Research Society Conference (FLAIRS-33)* (AAAI Press, 2020). https://arxiv.org/abs/2003.03072
49. Shang, J. et al. Learning Named Entity Tagger using Domain-Specific Dictionary. In *Proc. 2018 Conference on Empirical Methods in Natural Language Processing* 2054–2064 (Association for Computational Linguistics, 2018). https://doi.org/10.18653/v1/D18-1230
50. Liang, C. et al. BOND: BERT-Assisted Open-Domain Named Entity Recognition with Distant Supervision. In *Proc. 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20)* 1054–1064 (ACM, 2020). https://doi.org/10.1145/3394486.3403149
51. Lison, P., Barnes, J., Hubin, A. & Touileb, S. Named Entity Recognition without Labelled Data: A Weak Supervision Approach. In *Proc. 58th Annual Meeting of the Association for Computational Linguistics* 1518–1533 (Association for Computational Linguistics, 2020). https://doi.org/10.18653/v1/2020.acl-main.139
52. Wu, Q., Lin, Z., Karlsson, B. F., Lou, J.-G. & Huang, B. Single-/Multi-Source Cross-Lingual NER via Teacher-Student Learning on Unlabeled Data in Target Language. In *Proc. 58th Annual Meeting of the Association for Computational Linguistics* 6505–6514 (Association for Computational Linguistics, 2020). https://doi.org/10.18653/v1/2020.acl-main.581
53. Qu, X. et al. Distantly-Supervised Named Entity Recognition with Adaptive Teacher Learning and Fine-Grained Student Ensemble. *Proc. AAAI Conf. Artif. Intell.* **37**, 13501–13509 (2023). https://doi.org/10.1609/aaai.v37i11.26583
54. Du, J. et al. Self-training Improves Pre-training for Natural Language Understanding. In *Proc. 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* 5408–5418 (Association for Computational Linguistics, 2021). https://doi.org/10.18653/v1/2021.naacl-main.426
55. van Miltenburg, E., van der Lee, C. & Krahmer, E. Preregistering NLP research. In *Proc. 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* 613–623 (Association for Computational Linguistics, 2021). https://doi.org/10.18653/v1/2021.naacl-main.51
56. Lafferty, J., McCallum, A. & Pereira, F. Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data. In *Proc. 18th International Conference on Machine Learning (ICML 2001)* 282–289 (2001).
57. Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K. & Dyer, C. Neural Architectures for Named Entity Recognition. In *Proc. 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* 260–270 (Association for Computational Linguistics, 2016). https://doi.org/10.18653/v1/N16-1030
58. The Europe PMC Consortium. Europe PMC: a full-text literature database for the life sciences and platform for innovation. *Nucleic Acids Res.* **43**, D1042–D1048 (2015). https://doi.org/10.1093/nar/gku1061
59. Loshchilov, I. & Hutter, F. Decoupled Weight Decay Regularization. In *Proc. 7th International Conference on Learning Representations (ICLR 2019)* (2019). https://arxiv.org/abs/1711.05101
60. Nadeau, C. & Bengio, Y. Inference for the Generalization Error. *Mach. Learn.* **52**, 239–281 (2003). https://doi.org/10.1023/A:1024068626366

## Figure legends

**Figure 1. The names method, MR6 + names.** Three CRF-head teachers trained on the labeled
sentences tag a 40,000-sentence corpus; the mentions all three tag alike
with every marginal at 0.9 or more form the lexicon; the replacement pool
holds the training mentions and the lexicon; mention replacement writes
six copies of each training sentence with its mentions swapped; the
student trains on the gold sentences and the copies.

**Figure 2. The gain of MR6 + names on each held-out unit.** The
micro-F1 difference of MR6 + names (six-copy mention replacement with the
harvested names) from the CRF head (the baseline; red circles) and from
MR6 (six-copy mention replacement without them; blue squares), in
percentage points, each the difference of
the two systems' seed means on the unit: MuLMS's 7 subfields on the
decision seeds, MaterioMiner's 4 papers at both granularities and
SOFC-Exp's 5 folds, where MR6 was not run (the protocol put it at about
13 hours of one T4 GPU, on a corpus where the names method does not
pass; Supplementary Note 11). The shaded bands, drawn across the 11
units' rows, are the 95% Student-t intervals of the pooled differences
of the units shown, D5′ (red; the post hoc decision-seed reading of the
pooled decision D5) and D7 (blue; the names against MR6), with dashed lines at their means; the registered D5,
which takes MuLMS's units on all ten seeds, is given in red text below
the rows. The units carry no error bars.

**Figure 3. Recall and precision on the unseen test names.** Each point
pools the predictions over seeds and held-out units, 70 runs (7
subfields, 10 seeds) in **a)** and 40 (4 papers, 10 seeds) in **b)** and
**c)**: the CRF head (the baseline; black circles), mention replacement
with three and six copies (MR3, blue squares; MR6, cyan diamonds) and
MR6 with the harvested names (MR6 + names, red stars) on **a)** MuLMS's held-out
subfields, with the topic corpus and, orange plus sign, a mixed corpus;
**b)** MaterioMiner's held-out papers, fine-grained; and **c)** the same
papers, coarse-grained, with the fatigue corpus. Gray lines are iso-F1 curves; a gray arrow joins each
system to the one it builds on (the CRF head, MR3, MR6, then the names).
A point's label gives its F1 on the unseen names and, in parentheses, on
the seen ones; in **b)** and **c)** the labels carry the scores only.
Pooled rates carry no error bars. The mixed-corpus point uses seeds 300–900
(49 runs) and has no arrow, since MR6's point uses all ten seeds; on the
same seeds the two compare as in Table 4's note.

**Figure 4. The recall-matched control in each MuLMS subfield.**
**a)**–**g)** electrolysis, graphene, pemfc, poly, semi, sofc and steel
(decision seeds). Blue lines with dots: the recall over all test
entities of MR6 (six-copy mention replacement without the names) at each
bias D of the grid (the matching, pre-specified); red
lines with dots: its recall on the unseen test names the lexicon holds
(post hoc); both pooled over the seeds. Dashed lines of the same
colors: MR6 + names' values; dotted gray line: the matched bias D*. In
the three flagged subfields (**a)**, **c)** and **f)**) the blue curve peaks
below its dashed line; in every subfield the red curve stays below its
dashed line. Pooled rates carry no error bars.

## Tables

**Table 1.** The three corpora: documents and sentences, annotation, the
held-out unit with its test size, and the training set per unit.

| Corpus | Documents, sentences | Annotation | Held-out unit (test size) | Training per unit |
|---|---|---|---|---|
| MaterioMiner<sup>1</sup>, release v1.0.1, CC BY 4.0 | 4 papers on fatigue of metals (s11340, ncomms7164, s41467, met10111458: 41, 81, 201, 153 sentences) | the classes of the dataset's materials mechanics ontology (MMO) at two levels: fine-grained, its 27 classes with most entities, the other classes' entities relabeled O (1,328 entities; e.g. Alloy, Value, Mechanism, StateDescriptor, FatigueTestSpecimen, GrainBoundary); coarse-grained, 27 parent types (2,194 entities) | one paper (41–201 sentences) | the other three papers (275–435 sentences, of which 15% validate: 234–370 training sentences) |
| MuLMS<sup>3</sup>, Hugging Face `timo-pierre-schrader/MuLMS` at revision bc1c97d, CC BY-SA 4.0 | 50 articles from 7 materials subfields, 10,184 sentences: water electrolysis (electrolysis; 4 articles, 992 sentences), graphene (2, 351), polymer-electrolyte-membrane fuel cells (pemfc; 8, 1,638), polymers (poly; 11, 2,384), semiconductors (semi; 11, 2,040), solid oxide fuel cells (sofc; 4, 928), steel (10, 1,851) | 13 types (MAT, VALUE, PROPERTY, FORM, MEASUREMENT, CITE, SAMPLE, TECHNIQUE, DEV, INSTRUMENT, NUM, UNIT, RANGE); we keep the outermost of its nested entities, 31,952 of 46,584 | one subfield, every sentence of its articles (351–2,384) | 400 sentences of the other subfields' articles, taken in a shuffled order of those articles until 400 are reached (two or three articles, the last cut; a different set for each held-out subfield); 340 train and 60 validate, MaterioMiner's size by design |
| SOFC-Exp<sup>2</sup>, `boschresearch/sofc-exp_textmining_resources`, CC BY 4.0 | 45 articles on solid oxide fuel cells; 876 of their 9,466 sentences carry entity annotation (the ones that describe experiments) | 4 types: MATERIAL, VALUE, DEVICE, EXPERIMENT | 9 articles, one of 5 folds (137–248 sentences) | the other 36 articles (534–628 training and 94–111 validation sentences) |

**Table 2.** The twelve pre-registered decisions of the names route (D9
to D12 the follow-ups) in the order taken, with D5′, the post hoc
decision-seed reading of D5 and not a decision, in its own row. Columns: corpus and units (with
MuLMS's seeds), comparison, the reference's and the system's mean micro
F1 (for D11, recall), the mean paired difference with its 95% CI, the
two-sided P value of the same t-test (computed for reporting; Methods,
*Statistics and reproducibility*), and the verdict.

| | Corpus, units (MuLMS seeds) | Comparison | Reference | System | Δ [95% CI] | P | Verdict |
|---|---|---|---|---|---|---|---|
| D1 | MuLMS, 7 subfields (300–900) | MR6 + names vs CRF head | 48.54 | 51.73 | +3.19 [−1.13, +7.51] | 0.12 | no |
| D2 | MuLMS, 7 subfields (300–900) | MR6 + names vs MR3 | 50.81 | 51.73 | +0.92 [−1.63, +3.48] | 0.41 | no |
| D3 | MaterioMiner fine, 4 papers | MR6 + names vs CRF head | 46.46 | 50.72 | +4.27 [+1.39, +7.15] | 0.018 | passes |
| D4 | MaterioMiner coarse, 4 papers | MR6 + names vs CRF head | 58.60 | 60.38 | +1.77 [+0.67, +2.88] | 0.014 | passes |
| D5 | pooled: MuLMS 7 (0–900) + MaterioMiner fine 4 | MR6 + names vs CRF head | | | +3.58 [+0.95, +6.21], neither corpus below 0 | 0.013 | passes |
| D5′ | the same with MuLMS seeds 300–900 (the protocol's batch; post hoc reading) | | | | +3.58 [+1.04, +6.13] | 0.011 | above 0 (post hoc) |
| D6 | SOFC-Exp, 5 article folds | MR6 + names vs CRF head | 82.79 | 82.75 | −0.04 [−1.21, +1.14] | 0.94 | no |
| D7 | pooled, 11 units (300–900) | MR6 + names vs MR6 | | | +0.80 [−0.66, +2.26] | 0.25 | no |
| D8 | MuLMS, 7 subfields (300–900) | own-topic corpus vs mixed corpus | 51.91 | 51.73 | −0.18 [−2.01, +1.65] | 0.81 | no |
| D9 | MaterioMiner fine, 4 papers (follow-up) | multi-task + MR6 + names vs multi-task + MR6 | 52.99 | 53.95 | +0.96 [−0.01, +1.92] | 0.052 | no |
| D10 | MaterioMiner coarse, 4 papers (follow-up) | multi-task + MR6 + names vs multi-task + MR6 | 60.40 | 61.25 | +0.85 [−0.72, +2.42] | 0.18 | no |
| D11 | MuLMS, 7 subfields (300–900; follow-up) | MR6 + names vs MR6 at the recall-matched bias (3 of 7 subfields at the grid's end): recall on the unseen names the lexicon holds | 71.92 | 83.16 | +11.24 [+3.31, +19.18] | 0.013 | passes (as registered) |
| D12 | MuLMS, 7 subfields (300–900; follow-up) | MR6 + names vs MR6 at the recall-matched bias (3 of 7 subfields at the grid's end): micro F1 | 48.47 | 51.73 | +3.26 [−0.79, +7.31] | 0.096 | no |

The reference F1 on MuLMS is the 7-seed mean (48.54); the 10-seed mean
quoted in the Introduction is 48.66. D5 and D5′ are the two readings of one
protocol text (Methods, *Evaluation protocol*). Every column is micro F1 except D11's. For
D11 and D12 the reference is MR6 decoded at the bias that matches the
names' overall recall; in three subfields no bias on the grid (−2 to 5)
reaches it, and the protocol matched them at its end, D = 5, where MR6's
recall is 1.7 to 2.9 points lower. D11's columns are recall on the
unseen test names the lexicon holds, the mean over the subfields of rates
pooled over the seeds (Results, *What the names do*). Without the three
flagged subfields (pre-specified), D11 is +6.05 [−3.33, +15.44] (P =
0.13) and D12 −0.03 [−2.78, +2.72] (P = 0.97). D5's criterion was adopted on 20 October, when
mention replacement without names was already known to meet it (Methods,
*Evaluation protocol*).

**Table 3.** Every comparison among the four systems: over the 11
held-out units on both MuLMS seed bases (top) and over MaterioMiner's
four papers per granularity (bottom); the mean paired difference of micro
F1 with its 95% CI, for the 11 units the units above 0 and the exact
two-sided sign-flip P value, and the status of each entry. MR6's
per-corpus comparisons over MuLMS's seeds 0–900, pre-specified on 29
October, include its seeds 0–200, known from the screen; MR3's companion
decisions on MaterioMiner average in the three seeds of its screen, and
on the other seven (post hoc) the coarse-grained interval covers 0
(Discussion).

| Comparison, 11 units | MuLMS seeds 300–900: Δ micro F1 [95% CI] | units above 0 | sign-flip P | MuLMS seeds 0–900: Δ micro F1 [95% CI] | units above 0 | sign-flip P | status |
|---|---|---|---|---|---|---|---|
| MR3 − CRF head | +2.46 [+1.00, +3.91] | 11/11 | 0.001 | +2.35 [+0.94, +3.75] | 10/11 | 0.003 | post hoc |
| MR6 − MR3 | +0.33 [−0.25, +0.92] | 8/11 | 0.23 | +0.22 [−0.30, +0.74] | 8/11 | 0.37 | post hoc (per corpus pre-specified, 29 October) |
| MR6 − CRF head | +2.79 [+1.20, +4.38] | 10/11 | 0.002 | +2.56 [+0.94, +4.18] | 10/11 | 0.005 | post hoc (per corpus pre-specified, 29 October) |
| MR6 + names − MR6 | +0.80 [−0.66, +2.26] | 7/11 | 0.24 | +1.02 [−0.56, +2.59] | 6/11 | 0.18 | D7 on seeds 300–900; post hoc on 0–900 |
| MR6 + names − MR3 | +1.13 [−0.51, +2.76] | 7/11 | 0.16 | +1.23 [−0.22, +2.68] | 8/11 | 0.09 | post hoc (on MuLMS alone, D2) |
| MR6 + names − CRF head | +3.58 [+1.04, +6.13] | 10/11 | 0.007 | +3.58 [+0.95, +6.21] | 9/11 | 0.011 | D5 on seeds 0–900; D5′ a post hoc reading |

| MaterioMiner, 4 held-out papers (10 seeds) | fine-grained [95% CI] | coarse-grained [95% CI] | status |
|---|---|---|---|
| MR3 − CRF head | +2.77 [+1.51, +4.03] | +1.28 [+0.06, +2.51] | companion decisions (protocol of 16 October, 14:39), both passing |
| MR6 − MR3 | +0.20 [−1.47, +1.88] | −0.21 [−2.71, +2.29] | pre-specified, 29 October |
| MR6 − CRF head | +2.98 [+0.09, +5.86] | +1.07 [−1.82, +3.97] | pre-specified, 29 October |
| MR6 + names − MR6 | +1.29 [−2.51, +5.10] | +0.70 [−1.60, +3.00] | pre-specified, 29 October (fine-grained: D7's per-corpus condition) |
| MR6 + names − MR3 | +1.50 [−1.76, +4.75] | +0.49 [+0.06, +0.92] | pre-specified, 20 October (00:51) |
| MR6 + names − CRF head | +4.27 [+1.39, +7.15] | +1.77 [+0.67, +2.88] | D3; D4 |

**Table 4.** Recall, precision and F1 on the unseen test names, pooled
over seeds and MuLMS's seven held-out subfields (post hoc).

| MuLMS, unseen names (88% of the test entities); 70 runs per row (seeds 0–900), 49 for the mixed corpus (seeds 300–900) | recall | precision | F1 |
|---|---|---|---|
| CRF head | 39.4 | 55.8 | 46.15 |
| MR3 | 43.1 | 54.3 | 48.10 |
| MR6 | 43.7 | 53.9 | 48.28 |
| MR6 + names, the topic corpus | 47.3 | 50.9 | 49.01 |
| MR6 + names, a mixed corpus | 46.7 | 52.4 | 49.35 |

On the decision seeds alone (49 runs per row) the topic corpus's row reads
46.9 / 51.3 / 49.0, MR6's 43.9 / 54.2 / 48.5 and the CRF head's 39.1 /
55.8 / 46.0, so the two corpora's rows compare on the same seeds.

**Table 5.** Recall gains on the unseen test names by lexicon membership,
and the F1 gain on the names the lexicon holds (post hoc): per comparison, the gain on the names the lexicon holds and
on the others, their difference, paired over the units with 95% CIs, and
the units where the gain on the held names is the larger, and the F1
gain on the held names, each unit's F1 from its counts on them pooled
over the seeds, 2 × correct/(gold + predicted). The lexicon holds 22.2% of MuLMS's unseen
test names and 9.8% and 18.2% of MaterioMiner's (fine- and
coarse-grained). The mixed-corpus row is split by the topic corpus's
lexicon, which was the one rebuilt. MuLMS on the decision seeds,
MaterioMiner on 10 seeds.

| setting | comparison | recall gain, names in the lexicon [95% CI] | recall gain, names not in it [95% CI] | in − out [95% CI] | units with in > out | F1 gain, names in the lexicon [95% CI] |
|---|---|---|---|---|---|---|
| MuLMS, 7 subfields | MR6 + names − MR6 | +7.29 [+3.27, +11.30] | +1.12 [−0.78, +3.02] | +6.17 [+2.44, +9.90] | 7/7 | +1.13 [−1.85, +4.11] |
| MuLMS, 7 subfields | MR6 − CRF head | +3.39 [+0.92, +5.85] | +5.22 [+1.92, +8.52] | −1.84 [−5.27, +1.60] | 3/7 | +2.16 [+0.15, +4.16] |
| MuLMS, 7 subfields | MR6 + names (mixed corpus) − MR6 | +6.16 [+1.32, +11.01] | +1.31 [−1.07, +3.68] | +4.86 [+1.50, +8.21] | 7/7 | +1.18 [−1.63, +4.00] |
| MaterioMiner fine, 4 papers | MR6 + names − MR6 | +4.62 [−5.27, +14.50] | +0.29 [−4.23, +4.81] | +4.33 [−5.33, +13.98] | 3/4 | +1.96 [−6.54, +10.46] |
| MaterioMiner coarse, 4 papers | MR6 + names − MR6 | +0.36 [−1.11, +1.84] | +1.91 [−1.46, +5.28] | −1.54 [−3.60, +0.51] | 0/4 | +1.17 [−2.23, +4.58] |

**Table 6.** The names against the recall-matched control on MuLMS's
held-out subfields (follow-up; seeds 300–900): the names' recall over
all test entities pooled over the seeds, MR6's at bias 0, the matched
bias D*; recall on the unseen test names the lexicon holds for the
names, MR6 at D = 0 and the control at D*; and the names minus the
control at D* in that recall (D11), micro F1 (D12) and precision
(pre-specified), with the means over the subfields and their 95% CIs. In
a flagged subfield the names' recall exceeds MR6's at every bias of the
grid; it is matched at the grid's end, as the protocol fixed, and in
parentheses at the grid value of the nearest recall (post hoc). The last
two rows compare the names with MR6 at D = 0 (the recall post hoc, the
F1 D7's MuLMS part) and with MR6 at whichever bias gives it its highest
recall on the held names (post hoc).

| Subfield | names' recall | MR6's recall at D = 0 | D* | recall on the held names: names / MR6 at 0 / control at D* | Δ recall, held names | Δ micro F1 | Δ precision |
|---|---|---|---|---|---|---|---|
| electrolysis | 60.13 | 58.17 | 5 (flagged; 2.5) | 90.02 / 83.62 / 76.73 | +13.29 (+8.76) | +7.13 (+2.30) | +10.28 (+3.75) |
| graphene | 47.45 | 48.79 | −1.08 | 84.91 / 84.43 / 84.68 | +0.22 | −1.31 | −3.05 |
| pemfc | 51.74 | 49.23 | 5 (flagged; 3) | 84.92 / 78.09 / 69.33 | +15.59 (+11.18) | +6.43 (+3.28) | +11.03 (+6.50) |
| poly | 53.52 | 50.61 | 1.35 | 71.22 / 66.58 / 69.29 | +1.93 | −1.48 | −2.55 |
| semi | 31.78 | 26.82 | 2.47 | 81.36 / 73.58 / 68.96 | +12.40 | +2.20 | +6.90 |
| sofc | 50.36 | 46.55 | 5 (flagged; 2.5) | 85.32 / 71.23 / 59.72 | +25.61 (+16.23) | +9.39 (+4.15) | +16.93 (+8.82) |
| steel | 56.37 | 53.83 | 1.92 | 84.39 / 73.61 / 74.72 | +9.67 | +0.47 | +0.93 |
| mean, 7 subfields | | | | | +11.24 [+3.31, +19.18] (D11, passes as registered) | +3.26 [−0.79, +7.31] (D12, no) | +5.78 [−1.22, +12.79] |
| mean without the flagged, 4 subfields (pre-specified) | | | | | +6.05 [−3.33, +15.44] | −0.03 [−2.78, +2.72] | +0.56 [−6.74, +7.85] |
| mean, the flagged at the nearest recall, 7 subfields (post hoc) | | | | | +8.63 [+3.35, +13.90] | +1.37 [−0.66, +3.41] | +3.04 [−1.32, +7.41] |
| names − MR6 at D = 0, 7 subfields (recall post hoc; F1 D7's MuLMS part) | | | | | +7.29 [+3.27, +11.30] | +0.51 [−1.51, +2.53] | |
| names − MR6 at its best bias for the held names, 7 subfields (post hoc) | | | | | +6.52 [+2.15, +10.89] | | |
