# Cover letter

Dear Editors of *npj Computational Materials*,

I submit for your consideration the Article "Harvesting names from
unlabeled papers to augment materials named entity recognition on unseen
documents".

Materials text mining depends on named-entity recognizers trained on a
handful of annotated papers but run on new papers, most of whose names
those papers never annotate. The manuscript tests a cheap remedy. It
harvests the names that three baseline taggers confidently agree on in
40,000 sentences of open-access papers, and feeds them to mention
replacement, a standard data augmentation. On held-out documents the
combination improves on a MatSciBERT baseline by 3.6 F1 points pooled
over eleven held-out subfields and papers of two corpora, under a
criterion adopted mid-study when mention replacement alone already met
it, and on every held-out MaterioMiner paper at both annotation
granularities; a third corpus shows no gain. Three-copy mention
replacement alone gives about two thirds of that gain (post hoc), and
over six-copy mention replacement the names add 0.8 F1, not
established. On one corpus the harvested names raise recall on the
unseen names their lexicon holds, at some cost in precision (a post hoc
analysis), and a pre-registered test against a decoding bias matched in
recall passes, though only through three subfields that the bias could
not match.

The paper suits *npj Computational Materials* for three reasons:

1. It addresses a practical bottleneck of materials data extraction,
   generalization to documents unlike the annotated ones, with methods any
   group can apply to its own corpora in a few GPU hours.
2. It reports what the gains can and cannot be attributed to:
   - twelve decisions, each fixed in a protocol committed to a public
     repository before its results, with matched and recall-matched
     controls;
   - every comparison in every measure;
   - the dated protocol record, including a criterion I adopted mid-study,
     disclosed in full.

   Readers can therefore calibrate what to adopt. Mention replacement, the
   largest source of the gain, is the paper's recommendation.
3. Everything is reproducible. The repository holds the code, the
   protocols with their commit times, every results file with its seeds and
   predictions, and the analyses behind every number. The runs are
   deterministic, and every repeated run was identical.

**Related work.** A companion evaluation study from the same program
(H. Madi, in preparation) shares this paper's data, baselines and
protocol. It reports the gap between random splits and held-out documents,
seed variance, and the other additions tested. It has not been
submitted elsewhere, and no other related manuscript is in press or
under consideration elsewhere.

**Competing interests and funding.** I declare no competing interests.
The study received no funding; the GPU compute was Kaggle's free tier.

**Corresponding author.** Hatem Madi, American University of Beirut,
Beirut, Lebanon (hatem.madi15@gmail.com).

Sincerely,

Hatem Madi
