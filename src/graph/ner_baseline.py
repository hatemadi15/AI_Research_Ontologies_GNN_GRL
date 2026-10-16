"""
ner_baseline.py - MatSciBERT NER baseline (MaterioMiner protocol)

The only published results on MaterioMiner are NER results (Durmaz et al.,
Sci Data 2024): MatSciBERT fine-tuned on a random 65/15/20 sentence split,
five random initialisations, entity-level seqeval F1 (exact span and type):

    fine-grained (27 most frequent classes)   test F1 69.92
    coarse-grained (27 types)                 test F1 72.32

This script reproduces that protocol so the project has a number that is
comparable with the literature (the pipeline itself starts from gold spans).
The release ships the paper's split (--published-split); compare mean +/- std
over seeds there, and over several splits otherwise, not single runs.

  - Labels: fine = top-k classes by entity count (ties: token count, then
    name; the 27th/28th classes tie at 24 entities), other tags -> O;
    coarse = all types. The label list is saved with the results.
  - Split: random sentence split with --split-seed (65/15/20); --vary-split
    draws a new split per seed. --leave-one-paper-out trains on three papers
    and tests on the fourth (an out-of-distribution analogue of the 2025
    follow-up study, which reported ~60 F1 for MatSciBERT).
  - Model: m3rg-iitd/matscibert, first-subword labelling, AdamW + linear
    warm-up, gradient clipping, early stopping on validation F1. Variants:
    `--class-weights` (inverse square-root label frequency in the loss) or
    `--crf` (linear-chain CRF over the words with IOB2 constraints, trained by
    negative log-likelihood and decoded with Viterbi); `--crf --multi-task`
    trains one encoder for both granularities (below).
  - Metrics: seqeval micro F1 in default (conlleval) mode (primary) and
    strict IOB2 mode, plus per-class scores. The 2024 paper does not state
    its averaging ("averaging over the five random initializations and
    entity types"); the authors' 2025 follow-up uses micro F1. Macro and
    support-weighted F1 are therefore reported as well: on this data macro F1
    is several points lower than micro F1.

Results are written to <PROCESSED_DIR>/ner/ner_results_<granularity>[_<protocol>].json
after every run, and completed runs are skipped when re-run; re-running a
finished protocol only refreshes its `summary` (mean, population SD and 95%
Student-t CI over runs for micro, strict micro, macro and weighted F1). Runs
are resumed only if the file was written with the same labels, model, variant
and seed offset (and, with --save-predictions, stored their predictions). If a
results file holds runs that cannot be resumed, the script stops before any
training rather than replace them, unless --overwrite is given; smoke-test
files (--max-train) are replaced without it.

Training variants (ROADMAP P1) write their own files with a `_cw`, `_crf` or
`_crf_mt` suffix. `--tag NAME` appends a tag to the variant (`crf_m100`; a bare
`m100` for the baseline). `--seed-offset N` (needs a tag) adds N to every model
seed but keeps the data split and the run ids. `--save-predictions` stores per
run the corpus positions of the test sentences and their gold and predicted tags.

`--multi-task` (with `--crf` and both granularities; ROADMAP §4, week-2
protocol) trains one MatSciBERT encoder with a fine and a coarse linear head,
each with its own CRF, on the sum of the two NLLs. The splits are the
single-task ones, by corpus position. A sentence tokenised alike in both
corpora is one example with both tag sequences; the 15 that are not give two
examples, each labelled for one head only and never passed to the other
head's CRF. Each head is tested with the snapshot of its best validation F1.
One run writes both files (`_crf_mt`, single-task schema plus `multi_task`);
a run counts as done once both files hold it. `--granularity sofc_both` does the
same for SOFC-Exp's two levels: its slot types (`sofc_slot`, the 'fine' head)
and its entity types (`sofc`, the 'coarse' head), which tag the same sentences;
there the official split by article serves both (`--published-split`).

`--label-semantics` (with `--crf`; ROADMAP §4, week-3 protocol, variant 2)
replaces the linear token classifier by a label-semantic head: the score of B-X
(I-X) is the dot product of the token's state with W_B e_X (W_I e_X), e_X the
embedding of a description of X, and O has a learned vector (make_label_head).
The descriptions (ner_label_text.py: MMO definitions of the fine-grained
classes, the MMO classes annotated with each coarse-grained type) are embedded
in every run by the pre-trained encoder, before fine-tuning. Files `_crf_ls`,
compared with `crf`; the config stores the SHA-256 of the descriptions.

`--self-train CORPUS` (with `--crf`; ROADMAP §4, week-3 protocol, variant 3) trains
twice per run. The teacher is the plain CRF run. It tags a fixed sample of 2,000
sentences of CORPUS, one raw sentence per line: random.Random(0) shuffles the
sentence numbers and the first 2,000 are the sample, the same for every split and
seed. Each is split into words like the MaterioMiner files (corpus_words). A sentence
longer than the encoder's positions is skipped, not cut or replaced. Silver data are
the tagged sentences with an entity in which every word's Viterbi tag has a CRF
marginal of at least 0.9 (WordCRF.marginals). The student starts from the pre-trained
checkpoint with the teacher's seed and trains on gold + silver, each silver
sentence's NLL weighted 0.5; its epoch is chosen on the gold validation set and its
test scores are the run's. Files `_crf_st`, compared with `crf`; each run stores
`self_training` (sampled, skipped and silver sentences, the threshold, the teacher's
validation and test F1), the config the corpus path, SHA-256, sample size and silver
weight. Runs made from another corpus (SHA-256) are not resumed.

`--multi-task --label-semantics` (ROADMAP §4, week-4 protocol, item 2) gives each head of
the multi-task model the label-semantic head over its own granularity's descriptions,
embedded by the pre-trained encoder before fine-tuning. Files `_crf_mt_ls`; each stores the
SHA-256 of its head's descriptions.

`--self-train CORPUS --teachers N` (week-4 protocol, item 3) trains N CRF teachers per run,
with the model seeds of the seed offsets 0, 100, ..., 100 (N - 1), and each tags the sample
(`--self-train-sample 0`: the whole corpus). A sentence is silver if every teacher gives it
the same tags and it is silver for each teacher alone. Then every teacher seed trains a
student as above. One run writes N files, `_crf_st<N>_m<offset>` (the files of --tag
m<offset> --seed-offset <offset>), and each record stores the teachers' scores.

`--context N` (week-6 protocol) encodes each sentence between up to N subwords of the
neighbouring sentences of the same paper and part on each side (FLERT) and tags only the
sentence; SOFC-Exp takes them from its article texts (`context.json`). With `--multi-task`,
both heads read that input (files `_crf_mt_ctx`).

`--entity-bias D [D ...]` (with `--crf`; ROADMAP §4, week 7) decodes the test and
validation sentences again with D added to the emission of every tag but O, for each D
(recall calibration: the F1-optimal threshold of a calibrated model is half its F1, below
the 0.5 that a Viterbi decode applies). The run's own scores are those of D = 0. Per D it
stores the entity counts [correct, predicted, gold] of each test paper (`bias_test`) and of
the validation sentences (`bias_val`), and with --save-predictions the test predictions
(`test_pred_bias`); ner_bias.py chooses a D per held-out paper (files `_crf_bias`).
`--leave-two-papers-out` trains on two papers and tests on the other two, for each of the 6
pairs (folds `<paper>+<paper>`): with --entity-bias, the inner folds that choose the D of a
held-out paper without its labels.

On a CUDA device a run is deterministic (cuda_determinism) and every run stores its
`environment` (week-4 protocol: GPU runs are only compared with GPU runs).

`--compare` pairs every variant file with a reference file of the same
granularity and protocol, run by run (run_id), and writes ner_comparisons.json:
`crf_mt` and `crf_<tag>` with the `crf` file, `cw_<tag>` with `cw`, anything
else (`cw`, `crf`, a bare tag) with the baseline file.

Usage:
  python ner_baseline.py --granularity fine --seeds 0 1 2 3 4
  python ner_baseline.py --granularity coarse --epochs 2 --max-train 32   # smoke test
  python ner_baseline.py --granularity fine --vary-split --class-weights
  python ner_baseline.py --vary-split --crf --tag m100 --seed-offset 100 --save-predictions
  python ner_baseline.py --vary-split --crf --multi-task                  # fine + coarse heads
  python ner_baseline.py --vary-split --crf --label-semantics             # description head
  python ner_baseline.py --leave-one-paper-out --crf --tag m0 \\
      --self-train data/processed/corpus/fatigue_sentences.txt            # self-training
  python ner_baseline.py --vary-split --crf --multi-task --label-semantics --tag m0
  python ner_baseline.py --leave-one-paper-out --crf --teachers 3 --self-train-sample 0 \\
      --self-train data/processed/corpus/fatigue_sentences.txt            # 3 teachers
  python ner_baseline.py --leave-one-paper-out --crf --context 64 --tag m0   # document context
  python ner_baseline.py --leave-one-paper-out --crf --multi-task --context 64 --tag m0
  python ner_baseline.py --leave-one-paper-out --crf --multi-task --teachers 3 --students 10 \\
      --self-train data/processed/corpus/fatigue_sentences.txt --self-train-sample 0
  python ner_baseline.py --leave-one-paper-out --crf --entity-bias 0.5 1 2 --save-predictions
  python ner_baseline.py --leave-two-papers-out --crf --entity-bias 0.5 1 2   # inner folds
  python ner_baseline.py --granularity sofc_slot --published-split --crf    # SOFC-Exp slots
  python ner_baseline.py --granularity sofc_both --published-split --crf --multi-task
  python ner_baseline.py --compare                                        # paired comparison
"""

import argparse
import itertools
import json
import os
import random
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter
from types import SimpleNamespace

import conll
import dapt_mlm
import metrics
import ner_label_text
from config import PROCESSED_DIR, RAW_DIR

PAPER_F1 = {'fine': 69.92, 'coarse': 72.32,
            'sofc': None,       # SOFC-Exp (--granularity sofc): no published number of this kind
            'sofc_slot': None}
DEFAULT_MODEL = 'm3rg-iitd/matscibert'
# SOFC-Exp's two levels of types on the same sentences: the 4 entity types ('sofc') and the
# slot types ('sofc_slot'), of which those that its authors' evaluation scores are kept (the
# experiment-evoking word, the EXPERIMENT entity under another name, and the one
# interconnect_material become O)
SOFC = ('sofc', 'sofc_slot')
SOFC_SLOT_TYPES = ('anode_material', 'cathode_material', 'conductivity', 'current_density',
                   'degradation_rate', 'device', 'electrolyte_material', 'fuel_used',
                   'interlayer_material', 'open_circuit_voltage', 'power_density', 'resistance',
                   'support_material', 'thickness', 'time_of_operation', 'voltage',
                   'working_temperature')
# --granularity both / sofc_both: a dataset's two levels, the finer first (the multi-task
# heads 'fine' and 'coarse' of --multi-task)
GRANULARITY_PAIRS = {'both': ('fine', 'coarse'), 'sofc_both': ('sofc_slot', 'sofc')}


def load_split_corpus(granularity):
    """[(tokens, tags, doc_id)] for one granularity (files read separately); 'sofc' is the
    SOFC-Exp corpus (data/raw/dataset/sofc_exp, sofc_exp.py), one CoNLL file per article,
    and 'sofc_slot' its slot tags on the same sentences (data/raw/dataset/sofc_exp_slot)."""
    folder = {'fine': 'fine_grained_ner', 'coarse': 'coarse_grained_ner',
              'sofc': 'sofc_exp', 'sofc_slot': 'sofc_exp_slot'}[granularity]
    datadir = os.path.join(RAW_DIR, folder)
    data = []
    for fn in sorted(f for f in os.listdir(datadir) if f.endswith('.conll')):
        for tokens, tags in conll.read_conll_sentences(os.path.join(datadir, fn)):
            data.append((tokens, tags, fn))
    return data


# The published split (ROADMAP §4, week-5 amendment): the partitions of the MaterioMiner
# release, data/raw/dataset/published_split/ (see its README), checked by SHA-256
PUBLISHED_SPLIT = 'published_split'
PUBLISHED_PARTS = ('train', 'val', 'test')
PUBLISHED_SPLIT_SHA256 = {
    ('fine', 'train'): '6736fdb501a43317e4f2e6101bee5c911816f6dda813ae0807bd64c9f6cf8ae9',
    ('fine', 'val'): 'e874dcb5232bde370f2f659767254266ec98d48dcab86ade5de4271b324727ee',
    ('fine', 'test'): '043962bb9b022c42a474d5bc2b1fe3f7df799b90da0c6356dc1b73ecafad858a',
    ('coarse', 'train'): 'b0a3ba3146bebdbbc7c9937d474aded18cc2f81da950bc72879fa8ed9f22598b',
    ('coarse', 'val'): '7bf6965e2bfde17290f70dca5323d0f17674ede4c7fa17c05ab9d213495579db',
    ('coarse', 'test'): '3ecc82bc6e740a82866ae51b8f4271fead4e7b9bee9824692254fd5fbe6cb0a2',
}


def published_split(data, granularity):
    """([train, val, test positions in `data`], data with the release's tags) of the
    published split of a granularity (--published-split). For SOFC-Exp ('sofc', 'sofc_slot')
    it is the official split by article, with the corpus's own tags.

    The release files must have their recorded SHA-256, and each of their sentences must be
    a sentence of `data` (the same words), each used once, so that the three parts cover
    the corpus. Every sentence then takes the release's tags; the parts keep the order of
    the files.
    """
    if granularity in SOFC:
        # SOFC-Exp: the official split by article (split.json; train, dev and test)
        with open(os.path.join(RAW_DIR, 'sofc_exp', 'split.json')) as f:
            split = json.load(f)
        parts = [[i for i, (_, _, doc) in enumerate(data)
                  if split[doc.removesuffix('.conll')] == part] for part in ('train', 'dev', 'test')]
        return parts, list(data)
    import hashlib
    folder = os.path.join(RAW_DIR, PUBLISHED_SPLIT, f'{granularity}_grained_training')
    position = {}
    for i, (tokens, _, _) in enumerate(data):
        position.setdefault(tuple(tokens), []).append(i)
    data, parts, used = list(data), [], set()
    for part in PUBLISHED_PARTS:
        path = os.path.join(folder, f'{part}.txt')
        with open(path, 'rb') as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest != PUBLISHED_SPLIT_SHA256[granularity, part]:
            raise SystemExit(f'{path}: SHA-256 {digest}, not the recorded '
                             f'{PUBLISHED_SPLIT_SHA256[granularity, part]}')
        positions = []
        for tokens, tags in conll.read_conll_sentences(path):
            free = [i for i in position.get(tuple(tokens), []) if i not in used]
            if not free:
                raise SystemExit(f"{path}: a sentence that is not in the corpus, or used "
                                 f"twice: {' '.join(tokens)[:80]}")
            used.add(free[0])
            positions.append(free[0])
            data[free[0]] = (data[free[0]][0], list(tags), data[free[0]][2])
        parts.append(positions)
    if len(used) != len(data):
        raise SystemExit(f'the published split covers {len(used)} of {len(data)} sentences')
    return parts, data


def select_labels(data, top_k=None):
    """Entity types to keep: all, or the top_k by entity count."""
    entities = Counter()
    tokens = Counter()
    for _, tags, _ in data:
        for tag in tags:
            if tag.startswith('B-'):
                entities[tag[2:]] += 1
            if tag != 'O':
                tokens[tag[2:]] += 1
    ranked = sorted(entities, key=lambda t: (-entities[t], -tokens[t], t))
    if top_k is None or top_k >= len(ranked):
        return sorted(ranked), {}
    kept = ranked[:top_k]
    boundary = entities[kept[-1]]
    ties = [t for t in ranked if entities[t] == boundary]
    return sorted(kept), {'boundary_count': boundary, 'tied_classes': ties,
                          'kept_by_tiebreak': [t for t in ties if t in kept]}


def restrict_tags(tags, keep):
    """Map tags of classes outside `keep` to O and repair dangling I- tags."""
    out = []
    prev = 'O'
    for tag in tags:
        if tag == 'O' or tag[2:] not in keep:
            out.append('O')
        elif tag.startswith('I-') and prev[2:] != tag[2:]:
            out.append('B-' + tag[2:])
        else:
            out.append(tag)
        prev = out[-1]
    return out


def random_split(n, seed, ratios=(0.65, 0.15, 0.20)):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = round(ratios[0] * n)
    n_val = round(ratios[1] * n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


MR_P = 0.5                    # chance that --augment-mentions replaces a mention


def entity_spans(tags):
    """(start, end, type) of the entities of IOB2 tags; an I- tag that does not continue an
    entity of its type starts one, as seqeval's default mode reads it."""
    out, start, kind = [], None, None
    for i, tag in enumerate(list(tags) + ['O']):
        inside = start is not None and tag == f'I-{kind}'
        if start is not None and not inside:
            out.append((start, i, kind))
            start = None
        if tag != 'O' and not inside:
            start, kind = i, tag[2:]
    return out


def mention_replacements(sentences, k, seed, p=MR_P):
    """Mention replacement (Dai and Adel, 2020) for --augment-mentions: per sentence with an
    entity, k copies in which each mention is replaced, with probability p, by a mention of
    the same type drawn from all the mentions of `sentences` (frequent ones more often).
    Copies that change nothing are dropped. sentences: (words, tags); returns (words, tags).
    """
    rng = random.Random(seed)
    pool = {}
    for words, tags in sentences:
        for start, end, kind in entity_spans(tags):
            pool.setdefault(kind, []).append(tuple(words[start:end]))
    out = []
    for words, tags in sentences:
        spans = entity_spans(tags)
        if not spans:
            continue
        for _ in range(k):
            new_words, new_tags, last, changed = [], [], 0, False
            for start, end, kind in spans:
                mention = tuple(words[start:end])
                if rng.random() < p:
                    mention = rng.choice(pool[kind])
                    changed |= mention != tuple(words[start:end])
                new_words += list(words[last:start]) + list(mention)
                new_tags += list(tags[last:start]) + [f'B-{kind}'] + [f'I-{kind}'] * (len(mention) - 1)
                last = end
            if changed:
                out.append((new_words + list(words[last:]), new_tags + list(tags[last:])))
    return out


def part_contexts(data, part, full_texts=None):
    """Context sentences for --context: a (left, right) per position of `part`, in its order.

    The context of a sentence is the other sentences of the same paper within the same part
    (train, validation or test), as word lists, nearest first: left those before it, right
    those after it in corpus order. Other parts never serve as context, so no test sentence
    is seen in training.

    full_texts (SOFC-Exp, which annotates only some sentences of an article): {document stem:
    {'sentences': every sentence of its text, 'annotated': the positions of its annotated
    sentences}}. A document whose annotated sentences are all in `part` takes its context
    from the whole text, annotated sentences or not.
    """
    by_doc = {}
    for i in sorted(part):
        by_doc.setdefault(data[i][2], []).append(i)
    where = {i: (doc, k) for doc, idx in by_doc.items() for k, i in enumerate(idx)}
    whole = {}
    for doc, idx in by_doc.items():
        text = (full_texts or {}).get(doc.removesuffix('.conll'))
        if text and len(idx) == sum(d == doc for _, _, d in data):
            whole[doc] = text
    out = []
    for i in part:
        doc, k = where[i]
        if doc in whole:
            sentences, at = whole[doc]['sentences'], whole[doc]['annotated'][k]
            out.append((sentences[:at][::-1], sentences[at + 1:]))
            continue
        idx = by_doc[doc]
        out.append(([data[j][0] for j in reversed(idx[:k])], [data[j][0] for j in idx[k + 1:]]))
    return out


def load_full_texts(granularity):
    """The whole texts for document context (part_contexts): SOFC-Exp's context.json (both
    levels), or None (MaterioMiner, whose files hold every sentence of its papers)."""
    path = os.path.join(RAW_DIR, 'sofc_exp', 'context.json')
    if granularity not in SOFC or not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def context_words(tokenizer, left, right, budget):
    """(left words, right words) of a sentence's context (FLERT; Schweter and Akbik, 2020).

    Whole words of the context sentences (part_contexts), the nearest first, up to `budget`
    subwords on each side; the left words are returned in reading order.
    """
    def take(sentences, from_end):
        words, used = [], 0
        for sentence in sentences:
            for word in (reversed(sentence) if from_end else sentence):
                n = max(1, len(tokenizer([word], is_split_into_words=True,
                                         add_special_tokens=False)['input_ids']))
                if used + n > budget:
                    return words
                words.append(word)
                used += n
        return words
    return take(left, True)[::-1], take(right, False)


def encode(tokenizer, sentences, label2id, max_length=512, contexts=None, budget=0):
    """Tokenise pre-split words; label the first subword of each word.

    contexts (--context): a (left, right) per sentence from part_contexts. The words of up
    to `budget` subwords of each side (context_words; fewer if the sentence needs the room)
    surround the sentence unlabelled, so that only the sentence's words are scored.
    """
    features = []
    for k, (tokens, tags) in enumerate(sentences):
        left = right = []
        if contexts is not None:
            own = len(tokenizer(tokens, is_split_into_words=True, truncation=False)['input_ids'])
            left, right = context_words(tokenizer, *contexts[k],
                                        min(budget, max(0, (max_length - own) // 2)))
        first, last = len(left), len(left) + len(tokens)
        enc = tokenizer(left + list(tokens) + right, is_split_into_words=True, truncation=False)
        if len(enc['input_ids']) > max_length:
            raise ValueError(f"Sentence of {len(tokens)} words needs "
                             f"{len(enc['input_ids'])} subwords (> {max_length})")
        labels = []
        prev_word = None
        for word_id in enc.word_ids():
            if word_id is None or word_id == prev_word or not first <= word_id < last:
                labels.append(-100)
            else:
                labels.append(label2id[tags[word_id - first]])
            prev_word = word_id
        features.append({'input_ids': enc['input_ids'],
                         'attention_mask': enc['attention_mask'],
                         'labels': labels})
    return features


def batches(features, batch_size, pad_id, shuffle_seed=None):
    import torch
    order = list(range(len(features)))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [features[i] for i in order[start:start + batch_size]]
        width = max(len(f['input_ids']) for f in chunk)
        ids = [f['input_ids'] + [pad_id] * (width - len(f['input_ids'])) for f in chunk]
        mask = [f['attention_mask'] + [0] * (width - len(f['attention_mask'])) for f in chunk]
        labels = [f['labels'] + [-100] * (width - len(f['labels'])) for f in chunk]
        yield (torch.tensor(ids), torch.tensor(mask), torch.tensor(labels),
               order[start:start + batch_size])


def predict(model, features, batch_size, pad_id, id2label, device, crf=None):
    """Word-level predicted tags per sentence (first-subword positions).

    Without a CRF the tag of a word is the argmax at its first subword; with
    one it is the Viterbi path over the word-level emissions.
    """
    import torch
    model.eval()
    preds = [None] * len(features)
    with torch.no_grad():
        for ids, mask, labels, idx in batches(features, batch_size, pad_id):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            if crf is None:
                best = logits.argmax(-1).cpu()
                for row, i in enumerate(idx):
                    keep = labels[row] != -100
                    preds[i] = [id2label[int(p)] for p in best[row][keep]]
            else:
                emissions, _, word_mask = word_view(logits, labels)
                for i, path in zip(idx, crf.decode(emissions, word_mask)):
                    preds[i] = [id2label[p] for p in path]
    return preds


def entity_biases(args):
    """The biases a run decodes (--entity-bias): 0 and the given ones, sorted; [] without it."""
    if not hasattr(args, 'entity_bias'):
        return []
    return sorted({0.0, *(float(b) + 0.0 for b in args.entity_bias)})   # + 0.0: no -0.0


def bias_key(bias):
    """The key of a bias in the run fields of --entity-bias ('0', '0.5', '-1')."""
    return f'{bias:g}'


def biased_decodes(model, features, batch_size, pad_id, id2label, device, crf, biases):
    """{bias: word-level predicted tags per sentence} (--entity-bias): the CRF's Viterbi
    paths after adding `bias` to the emission of every tag but O, the encoder run once per
    batch. Bias 0 adds nothing, so its paths are predict()'s."""
    import torch
    model.eval()
    preds = {b: [None] * len(features) for b in biases}
    shift = torch.tensor([0.0 if id2label[i] == 'O' else 1.0 for i in range(len(id2label))])
    with torch.no_grad():
        for ids, mask, labels, idx in batches(features, batch_size, pad_id):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            emissions, _, word_mask = word_view(logits, labels)
            step = shift.to(device=emissions.device, dtype=emissions.dtype)
            for b in biases:
                for i, path in zip(idx, crf.decode(emissions + b * step, word_mask)):
                    preds[b][i] = [id2label[p] for p in path]
    return preds


def entity_counts(gold_tags, pred_tags):
    """[correct, predicted, gold] entities of tag sequences, counted as seqeval's default
    mode counts them for micro F1 (score): F1 = 200 * correct / (predicted + gold)."""
    from seqeval.metrics.sequence_labeling import get_entities
    gold, pred = set(get_entities(gold_tags)), set(get_entities(pred_tags))
    return [len(gold & pred), len(pred), len(gold)]


def bias_fields(biases, test, test_preds, val, val_preds, save_predictions=False):
    """The fields --entity-bias adds to a run, per bias (bias_key): the entity counts of each
    test document (bias_test: {bias: {doc: counts}}) and of the validation sentences together
    (bias_val: {bias: counts}); with save_predictions the test predictions (test_pred_bias:
    {bias: space-joined tags per test sentence, in test order}). test and val are lists of
    (tokens, tags, doc); test_preds and val_preds are biased_decodes' results for them."""
    docs = list(dict.fromkeys(d for _, _, d in test))
    fields = {'bias_test': {}, 'bias_val': {}}
    for b in biases:
        key = bias_key(b)
        fields['bias_test'][key] = {
            doc: entity_counts([g for _, g, d in test if d == doc],
                               [p for (_, _, d), p in zip(test, test_preds[b]) if d == doc])
            for doc in docs}
        fields['bias_val'][key] = entity_counts([g for _, g, _ in val], val_preds[b])
    if save_predictions:
        fields['test_pred_bias'] = {bias_key(b): [' '.join(p) for p in test_preds[b]]
                                    for b in biases}
    return fields


def score(gold_tags, pred_tags):
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    from seqeval.scheme import IOB2
    return {
        'f1': round(100 * f1_score(gold_tags, pred_tags), 2),
        'precision': round(100 * precision_score(gold_tags, pred_tags), 2),
        'recall': round(100 * recall_score(gold_tags, pred_tags), 2),
        'f1_strict': round(100 * f1_score(gold_tags, pred_tags, mode='strict',
                                          scheme=IOB2), 2),
        'per_class': {k: {m: round(float(v), 4) for m, v in d.items()}
                      for k, d in classification_report(
                          gold_tags, pred_tags, output_dict=True,
                          zero_division=0).items()},
    }


def prediction_fields(test_index, test, test_pred):
    """Per-sentence test predictions of a run (--save-predictions).

    test_index: positions of the test sentences in the corpus; test_gold and
    test_pred: their gold and predicted IOB2 tags, joined by single spaces.
    """
    return {'test_index': [int(i) for i in test_index],
            'test_gold': [' '.join(tags) for _, tags, _ in test],
            'test_pred': [' '.join(tags) for tags in test_pred]}


# CRF head (--crf). Transitions start at zero and need far larger steps than the
# encoder, so they get their own learning rate (100x the default encoder rate),
# fixed in advance rather than tuned.
CRF_LR = 5e-3
CRF_FORBIDDEN = -1e4   # added to transitions that break IOB2


def bio_allowed(label_list):
    """IOB2 constraints: allowed[i][j] for label i -> label j, and allowed starts.

    I-X may only follow B-X or I-X, and a sentence may not start with I-X.
    """
    def ok(prev, cur):
        return not cur.startswith('I-') or (prev != 'O' and prev[2:] == cur[2:])
    allowed = [[ok(prev, cur) for cur in label_list] for prev in label_list]
    start = [not cur.startswith('I-') for cur in label_list]
    return allowed, start


def word_view(logits, labels):
    """Word-level emissions, gold tags and mask from subword logits.

    Words are the first-subword positions (labels != -100). Returns emissions
    (B, W, L), tags (B, W) and a left-aligned boolean mask (B, W).
    """
    import torch
    keep = labels != -100
    n_words = keep.sum(1)
    width = max(int(n_words.max()), 1)
    idx = torch.zeros(labels.size(0), width, dtype=torch.long)
    for b in range(labels.size(0)):
        pos = keep[b].nonzero(as_tuple=True)[0]
        idx[b, :len(pos)] = pos
    mask = torch.arange(width).unsqueeze(0) < n_words.unsqueeze(1)
    idx, mask = idx.to(logits.device), mask.to(logits.device)
    emissions = logits.gather(1, idx.unsqueeze(2).expand(-1, -1, logits.size(2)))
    tags = labels.to(logits.device).gather(1, idx).clamp(min=0) * mask
    return emissions, tags, mask


_CRF_CLASS = None


def make_crf(label_list):
    """Linear-chain CRF over word-level emissions with IOB2 constraints.

    The class is defined on first use so that importing this module does not
    need torch.
    """
    global _CRF_CLASS
    if _CRF_CLASS is None:
        _CRF_CLASS = _define_crf()
    return _CRF_CLASS(label_list)


def _define_crf():
    import torch

    class WordCRF(torch.nn.Module):
        def __init__(self, label_list):
            super().__init__()
            n = len(label_list)
            allowed, start_ok = bio_allowed(label_list)
            self.transitions = torch.nn.Parameter(torch.zeros(n, n))
            self.start = torch.nn.Parameter(torch.zeros(n))
            self.end = torch.nn.Parameter(torch.zeros(n))
            self.register_buffer('trans_penalty',
                                 CRF_FORBIDDEN * (~torch.tensor(allowed)).float())
            self.register_buffer('start_penalty',
                                 CRF_FORBIDDEN * (~torch.tensor(start_ok)).float())

        def scores(self):
            """Transition, start and end scores with the IOB2 penalties applied."""
            return (self.transitions + self.trans_penalty,
                    self.start + self.start_penalty, self.end)

        def nll(self, emissions, tags, mask, weights=None):
            """Mean negative log-likelihood of the gold tag sequences.

            emissions (B, W, L), tags (B, W), mask (B, W): True on words,
            left-aligned, at least one word per sequence. weights (B,), optional:
            the loss is then sum_i weights[i] * NLL_i / B (--self-train: silver
            sentences 0.5, gold ones 1, which keep their share of the mean).
            """
            trans, start, end = self.scores()
            gold = start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, :1]).squeeze(1)
            alpha = start + emissions[:, 0]
            for t in range(1, emissions.size(1)):
                m = mask[:, t]
                emit = emissions[:, t]
                step = (trans[tags[:, t - 1], tags[:, t]]
                        + emit.gather(1, tags[:, t:t + 1]).squeeze(1))
                gold = gold + step * m.to(step.dtype)
                nxt = torch.logsumexp(alpha.unsqueeze(2) + trans + emit.unsqueeze(1), dim=1)
                alpha = torch.where(m.unsqueeze(1), nxt, alpha)
            last = tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
            gold = gold + end[last]
            log_z = torch.logsumexp(alpha + end, dim=1)
            nlls = log_z - gold
            return nlls.mean() if weights is None else (nlls * weights).mean()

        def log_partition(self, emissions, mask):
            """log Z per sequence (B,): the forward algorithm of nll.

            nll keeps its own copy of the recursion, so that training is unchanged.
            """
            trans, start, end = self.scores()
            alpha = start + emissions[:, 0]
            for t in range(1, emissions.size(1)):
                nxt = torch.logsumexp(alpha.unsqueeze(2) + trans + emissions[:, t].unsqueeze(1),
                                      dim=1)
                alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
            return torch.logsumexp(alpha + end, dim=1)

        def marginals(self, emissions, mask):
            """(B, W, L): the probability of each tag at each word under the CRF, 0 on padding.

            The gradient of log Z with respect to the emissions (forward-backward by
            autograd). Tags that the IOB2 penalties forbid get a probability of about 0.
            """
            with torch.enable_grad():
                emissions = emissions.detach().requires_grad_()
                log_z = self.log_partition(emissions, mask)
                return torch.autograd.grad(log_z.sum(), emissions)[0]

        def decode(self, emissions, mask):
            """Viterbi label-id paths, one list per sequence."""
            trans, start, end = self.scores()
            score = start + emissions[:, 0]
            backpointers = []
            for t in range(1, emissions.size(1)):
                best, idx = (score.unsqueeze(2) + trans).max(dim=1)
                score = torch.where(mask[:, t].unsqueeze(1), best + emissions[:, t], score)
                backpointers.append(idx)
            score = score + end
            paths = []
            for b, length in enumerate(mask.sum(1).tolist()):
                path = [int(score[b].argmax())]
                for t in range(length - 1, 0, -1):
                    path.append(int(backpointers[t - 1][b, path[-1]]))
                paths.append(path[::-1])
            return paths

    return WordCRF


def class_weights(features, n_labels):
    """Inverse square-root frequency weights over first-subword labels."""
    import torch
    counts = Counter(lab for f in features for lab in f['labels'] if lab != -100)
    weights = torch.ones(n_labels)
    for lab in range(n_labels):
        weights[lab] = 1.0 / max(counts.get(lab, 1), 1) ** 0.5
    return weights / weights.mean()


# Label-semantic head (--label-semantics): descriptions are cut to this many tokens
DESCRIPTION_MAX_TOKENS = 128


def label_descriptions(granularity, labels):
    """Descriptions of `labels`, in their order (ner_label_text).

    Fine: from the MMO definitions; coarse: from the fine-grained classes of the
    corpus annotated with each type.
    """
    if granularity == 'fine':
        return ner_label_text.fine_descriptions(labels)
    return ner_label_text.coarse_descriptions(labels, load_split_corpus('fine'),
                                              load_split_corpus('coarse'))


def embed_descriptions(encoder, tokenizer, texts, device, max_tokens=DESCRIPTION_MAX_TOKENS):
    """(len(texts), hidden): the mean last-layer state of `encoder` over each text's tokens.

    Each text is encoded alone, cut to max_tokens (special tokens included), in eval
    mode and without gradient; the special tokens are left out of the mean.
    """
    import torch
    training = encoder.training
    encoder.eval()
    rows = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=max_tokens,
                            return_special_tokens_mask=True, return_tensors='pt')
            states = encoder(input_ids=enc['input_ids'].to(device),
                             attention_mask=enc['attention_mask'].to(device)).last_hidden_state
            rows.append(states[0][enc['special_tokens_mask'][0].to(device) == 0].mean(0))
    encoder.train(training)
    return torch.stack(rows)


_HEAD_CLASS = None


def make_label_head(embeddings, config):
    """Label-semantic token head over (n_labels, hidden) description embeddings.

    It gives the 1 + 2 * n_labels scores of the tag list. Defined on first use, like
    the CRF, so that importing this module does not need torch.
    """
    global _HEAD_CLASS
    if _HEAD_CLASS is None:
        _HEAD_CLASS = _define_label_head()
    return _HEAD_CLASS(embeddings, config.hidden_size, config.initializer_range)


def _define_label_head():
    import torch

    class LabelSemanticHead(torch.nn.Module):
        """Tag scores in the order of the tag list: O, then B-X and I-X for each label X.

        score(t, O) = h_t . v_O, score(t, B-X) = h_t . (W_B e_X) and score(t, I-X) =
        h_t . (W_I e_X), each plus a learned bias per tag. e_X is the embedding of the
        description of X, a frozen buffer (`descriptions`); W_B and W_I (hidden x
        hidden) and v_O are learned.
        """

        def __init__(self, embeddings, hidden_size, std):
            super().__init__()
            n_labels, dim = embeddings.shape
            # Initialisation: one factor for all labels scales the embeddings so that the mean
            # of |e_X|^2 is 1 (a linear map absorbs a common factor), and W_B, W_I and v_O start
            # as N(0, std^2), std the initializer_range of the default classifier, with zero
            # bias. The entries of W_B e_X are then N(0, std^2 |e_X|^2), on average over the
            # labels those of a row of the default classifier, so the initial logits have its
            # magnitude (MatSciBERT: SD 0.43 fine-grained, 0.41 coarse-grained, against 0.48).
            # Scaling the embeddings rather than shrinking W keeps the weights at the
            # classifier's scale, relative to which AdamW's steps (about lr) are as small.
            rms = embeddings.pow(2).sum(1).mean().sqrt()
            self.register_buffer('descriptions', embeddings / rms)
            self.w_b = torch.nn.Parameter(torch.empty(hidden_size, dim))
            self.w_i = torch.nn.Parameter(torch.empty(hidden_size, dim))
            self.v_o = torch.nn.Parameter(torch.empty(hidden_size))
            self.bias = torch.nn.Parameter(torch.zeros(1 + 2 * n_labels))
            for weight in (self.w_b, self.w_i, self.v_o):
                torch.nn.init.normal_(weight, mean=0.0, std=std)

        def tag_vectors(self):
            """(1 + 2 * n_labels, hidden): v_O, then W_B e_X and W_I e_X for each label X."""
            e = self.descriptions
            pairs = torch.stack([e @ self.w_b.T, e @ self.w_i.T], dim=1)
            return torch.cat([self.v_o.unsqueeze(0), pairs.flatten(0, 1)])

        def forward(self, hidden):
            return torch.nn.functional.linear(hidden, self.tag_vectors(), self.bias)

    return LabelSemanticHead


# Self-training (--self-train), fixed by the week-3 protocol
SELF_TRAIN_SAMPLE = 2000      # corpus sentences that the teacher tags
SELF_TRAIN_SEED = 0           # random.Random seed of the sample: the same for every run
SILVER_THRESHOLD = 0.9        # least CRF marginal of every word's tag in a silver sentence
SILVER_WEIGHT = 0.5           # NLL weight of a silver sentence (gold: 1)

# Words as in the MaterioMiner files: a run of letters and digits in which '-' or an
# apostrophe may join two letters (displacement-controlled, Young's), '.' two letters or
# two digits (i.e, 0.6) and ',' two digits (5,6), then '%' after a digit (4.1%). Any
# other non-space character is a word of its own: brackets, sentence punctuation, '/',
# '°', and a '-' next to a digit or a bracket (10 - kN, - 35, ( Ni ) - base).
WORD_RE = re.compile(r"[^\W_]+(?:(?:(?<=[^\W\d_])[-'.](?=[^\W\d_])|(?<=\d)[.,](?=\d))[^\W_]+)*"
                     r"(?:(?<=\d)%)?|\S")
QUOTES = {'\u2018': "'", '\u2019': "'"}


def corpus_words(text):
    """A raw corpus sentence split into words like the MaterioMiner files (WORD_RE).

    The text is normalised first, as in dapt_mlm.normalise but not lowercased: Unicode
    NFKC; every dash (category Pd) and the minus sign U+2212 written '-', and the quotes
    U+2018 and U+2019 "'" (the only dash and apostrophe of the MaterioMiner text); and
    the characters that the BERT tokenizer drops (category C*, e.g. a soft hyphen)
    removed, so that they split no word.
    """
    chars = []
    for ch in unicodedata.normalize('NFKC', text):
        category = unicodedata.category(ch)
        if category[0] == 'C' and not ch.isspace():
            continue
        chars.append('-' if category == 'Pd' or ch == '\u2212' else QUOTES.get(ch, ch))
    return WORD_RE.findall(''.join(chars))


def self_training_corpus(path, size=SELF_TRAIN_SAMPLE, seed=SELF_TRAIN_SEED):
    """The fixed sample of a --self-train corpus as word lists, and its config entries.

    One raw sentence per line, blank lines skipped (dapt_mlm.read_sentences).
    random.Random(seed) shuffles the sentence numbers and the first `size` make the
    sample, in corpus order: independent of the split and the model seed, and drawn
    before any sentence is skipped (fit_sample). Size 0 takes the whole corpus. The
    config entries are the corpus path (relative to the project root when inside it), its
    SHA-256, the sample size and the silver weight.
    """
    sentences, _ = dapt_mlm.read_sentences(path)
    if not sentences:
        raise SystemExit(f'{path}: no sentences to self-train on')
    order = list(range(len(sentences)))
    random.Random(seed).shuffle(order)
    size = size or len(sentences)          # 0: the whole corpus (--self-train-sample 0)
    sample = [corpus_words(sentences[i]) for i in sorted(order[:size])]
    return sample, {'self_train': dapt_mlm.display_path(path),
                    'self_train_sha256': dapt_mlm.file_sha256(path),
                    'self_train_sample_size': size, 'self_train_silver_weight': SILVER_WEIGHT}


def fit_sample(tokenizer, sample, max_length):
    """The sample sentences that the encoder can tag whole, and how many were skipped.

    A sentence of more than max_length subwords ([CLS] and [SEP] included) is skipped,
    not cut, and not replaced by another. A word without subwords (a lone combining
    mark, say) is left out of its sentence, since it cannot carry a tag; a sentence left
    without words is skipped as well.
    """
    kept = []
    for words in sample:
        enc = tokenizer(words, is_split_into_words=True, truncation=False)
        present = sorted({i for i in enc.word_ids() if i is not None})
        if present and len(enc['input_ids']) <= max_length:
            kept.append([words[i] for i in present])
    return kept, len(sample) - len(kept)


def teacher_tags(tagger, sentences, batch_size, device):
    """Viterbi tags of word lists by a trained CRF tagger, with their CRF marginals.

    tagger: model, crf, tokenizer, label2id and id2label (train_one with keep_tagger).
    Returns [(tags, marginals)] in the order of `sentences`; marginals[t] is the
    probability of tags[t] at word t.
    """
    import torch
    model, crf = tagger.model, tagger.crf
    features = encode(tagger.tokenizer, [(words, ['O'] * len(words)) for words in sentences],
                      tagger.label2id, model.config.max_position_embeddings)
    model.eval()
    out = [None] * len(features)
    with torch.no_grad():
        for ids, mask, labels, idx in batches(features, batch_size, tagger.tokenizer.pad_token_id):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            emissions, _, word_mask = word_view(logits, labels)
            marginals = crf.marginals(emissions, word_mask)
            for row, (i, path) in enumerate(zip(idx, crf.decode(emissions, word_mask))):
                out[i] = ([tagger.id2label[p] for p in path],
                          [float(marginals[row, t, p]) for t, p in enumerate(path)])
    return out


def is_silver(tags, marginals):
    """Silver sentence: an entity, and every word's tag with a marginal >= SILVER_THRESHOLD."""
    return any(tag.startswith('B-') for tag in tags) and min(marginals) >= SILVER_THRESHOLD


def train_one(seed, train, val, test, labels, args, device, test_index=None, descriptions=None,
              silver=None, keep_tagger=False, contexts=None, augmented=None):
    """Fine-tune and test once; `seed` is the model seed (--seed-offset included).

    test_index (positions of `test` in the corpus) is stored with the
    predictions when --save-predictions is set. descriptions (--label-semantics):
    a text per label of `labels`; the token classifier is then a label-semantic
    head over their embeddings by the pre-trained encoder (make_label_head).
    silver (--self-train, with the CRF): (words, tags) sentences trained on with the
    gold ones, each sentence's NLL weighted SILVER_WEIGHT. keep_tagger: return
    (result, tagger), the tagger holding the model with its best weights, the CRF,
    tokenizer, label2id and id2label (teacher_tags). contexts (--context): the
    part_contexts of train, val and test ({'train': ..., 'val': ..., 'test': ...}),
    encoded around each sentence with up to args.context subwords per side. augmented
    (--augment-mentions): (words, tags) sentences trained on as gold ones.
    """
    import torch
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup, set_seed)

    set_seed(seed)
    label_list = ['O'] + [f'{p}-{t}' for t in labels for p in ('B', 'I')]
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    assert tokenizer.is_fast, 'a fast tokenizer is required for word_ids()'
    # float32 whatever the checkpoint holds: transformers 5 loads a model in the dtype of its
    # weights, and DeBERTa-v3's are float16, where training gives a NaN loss
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(label_list), id2label=id2label, label2id=label2id,
        dtype=torch.float32).to(device)
    if descriptions is not None:
        # embedded before any fine-tuning; the head replaces the linear classifier
        assert len(descriptions) == len(labels), 'one description per label'
        embeddings = embed_descriptions(model.base_model, tokenizer, descriptions, device)
        model.classifier = make_label_head(embeddings, model.config).to(device)
    pad_id = tokenizer.pad_token_id
    crf = make_crf(label_list).to(device) if getattr(args, 'crf', False) else None

    def encode_part(name, part):
        sentences = [(t, g) for t, g, _ in part]
        if contexts is None:
            return encode(tokenizer, sentences, label2id)
        return encode(tokenizer, sentences, label2id, model.config.max_position_embeddings,
                      contexts[name], args.context)

    f_train = encode_part('train', train)
    if silver:
        assert crf is not None, 'silver sentences are weighted in the CRF loss'
        # the gold sentences, then the silver ones; batches() shuffles them together
        weight = [1.0] * len(f_train) + [SILVER_WEIGHT] * len(silver)
        f_train += encode(tokenizer, silver, label2id, model.config.max_position_embeddings)
    if augmented:
        assert not silver, 'augmented sentences are gold ones, never silver'
        f_train += encode(tokenizer, augmented, label2id, model.config.max_position_embeddings)
    f_val = encode_part('val', val)
    f_test = encode_part('test', test)

    loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weights(f_train, len(label_list)).to(device) if args.class_weights else None,
        ignore_index=-100)
    if crf is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(
            [{'params': model.parameters()},
             {'params': crf.parameters(), 'lr': CRF_LR, 'weight_decay': 0.0}],
            lr=args.lr, weight_decay=0.01)
    params = list(model.parameters()) + (list(crf.parameters()) if crf is not None else [])
    steps = args.epochs * ((len(f_train) + args.batch_size - 1) // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)

    def snapshot(module):
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    best = (-1.0, None, -1)
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for ids, mask, lab, idx in batches(f_train, args.batch_size, pad_id,
                                           shuffle_seed=seed * 1000 + epoch):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            if crf is None:
                loss = loss_fn(logits.view(-1, logits.size(-1)), lab.to(device).view(-1))
            elif silver:
                loss = crf.nll(*word_view(logits, lab),
                               weights=torch.tensor([weight[i] for i in idx], device=device))
            else:
                loss = crf.nll(*word_view(logits, lab))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += loss.item()
        val_pred = predict(model, f_val, args.batch_size, pad_id, id2label, device, crf)
        val_f1 = score([g for _, g, _ in val], val_pred)['f1']
        history.append({'epoch': epoch, 'train_loss': round(total, 4), 'val_f1': val_f1})
        print(f"    seed {seed} epoch {epoch}: loss {total:.3f} val F1 {val_f1:.2f}", flush=True)
        if val_f1 > best[0]:
            best = (val_f1, (snapshot(model), snapshot(crf) if crf is not None else None), epoch)
        elif epoch - best[2] >= args.patience:
            break

    model.load_state_dict(best[1][0])
    if crf is not None:
        crf.load_state_dict(best[1][1])
    biases = entity_biases(args)
    if biases:
        decodes = biased_decodes(model, f_test, args.batch_size, pad_id, id2label, device, crf,
                                 biases)
        test_pred = decodes[0.0]
    else:
        test_pred = predict(model, f_test, args.batch_size, pad_id, id2label, device, crf)
    result = score([g for _, g, _ in test], test_pred)
    result.update({'seed': seed, 'best_epoch': best[2], 'val_f1': best[0],
                   'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
                   'history': history})
    if getattr(args, 'save_predictions', False):
        result.update(prediction_fields(test_index, test, test_pred))
    if biases:
        result.update(bias_fields(
            biases, test, decodes, val,
            biased_decodes(model, f_val, args.batch_size, pad_id, id2label, device, crf, biases),
            getattr(args, 'save_predictions', False)))
    if keep_tagger:
        return result, SimpleNamespace(model=model, crf=crf, tokenizer=tokenizer,
                                       label2id=label2id, id2label=id2label)
    return result


def train_self_training(seed, train, val, test, labels, args, device, sample, test_index=None):
    """One --self-train run: teacher, silver data and student; returns the student's record.

    The teacher is train_one on the run's data, the plain --crf run. It tags the sample
    (word lists; fit_sample, teacher_tags) and is_silver picks the silver sentences. The
    student is train_one with the same seed, hence the same initialisation, on the gold
    data plus the silver sentences. The record is the student's, plus `self_training`:
    n_sampled, n_skipped (fit_sample), n_silver, marginal_threshold, teacher_val_f1 and
    teacher_test_f1 (micro F1).
    """
    teacher_args = SimpleNamespace(**{**vars(args), 'save_predictions': False})
    teacher, tagger = train_one(seed, train, val, test, labels, teacher_args, device,
                                keep_tagger=True)
    sentences, n_skipped = fit_sample(tagger.tokenizer, sample,
                                      tagger.model.config.max_position_embeddings)
    tagged = teacher_tags(tagger, sentences, args.batch_size, device)
    del tagger                     # the teacher is not needed any more
    silver = [(words, tags) for words, (tags, marginals) in zip(sentences, tagged)
              if is_silver(tags, marginals)]
    print(f"    teacher: val F1 {teacher['val_f1']:.2f}, test F1 {teacher['f1']:.2f}; "
          f"{len(silver)} silver of {len(sentences)} tagged sentences ({n_skipped} of "
          f"{len(sample)} sampled skipped); student:", flush=True)
    student = train_one(seed, train, val, test, labels, args, device, test_index=test_index,
                        silver=silver)
    student['self_training'] = {
        'n_sampled': len(sample), 'n_skipped': n_skipped, 'n_silver': len(silver),
        'marginal_threshold': SILVER_THRESHOLD, 'teacher_val_f1': teacher['val_f1'],
        'teacher_test_f1': teacher['f1']}
    return student


def teacher_offsets(n):
    """Seed offsets of --teachers n: 0, 100, ..., 100 (n - 1), as the seed families."""
    return [100 * k for k in range(n)]


def unanimous_silver(tagged):
    """Silver with several teachers: [(tags, least marginal)] of one sentence, per teacher.

    Every teacher gives the same tags, and the sentence is silver for each of them alone
    (an entity, every tag with a marginal of at least SILVER_THRESHOLD; is_silver).
    """
    first = tagged[0][0]
    return (all(tags == first for tags, _ in tagged)
            and all(is_silver(tags, [least]) for tags, least in tagged))


def train_with_teachers(seed, offsets, train, val, test, labels, args, device, sample,
                        test_index=None, student_offsets=None):
    """One --self-train --teachers run: {offset: the record of the student of that seed}.

    The teachers are train_one on the run's data with the model seeds seed + offset, the
    plain --crf runs of those seeds. Each tags the sample (fit_sample, teacher_tags); a
    sentence is silver if unanimous_silver. Then each seed of `student_offsets` (default:
    the teachers' offsets; --students) trains a student, train_one with that seed on the
    gold data plus the silver sentences, as in train_self_training. Each record adds
    `self_training`: n_sampled, n_skipped, n_agreed (sentences that all teachers tag
    alike), n_silver, marginal_threshold, and per teacher (in offset order) its seed,
    val_f1, test_f1, test_f1_strict, test_macro_f1, best_epoch and n_silver_alone (its
    silver sentences as a single teacher).
    """
    teacher_args = SimpleNamespace(**{**vars(args), 'save_predictions': False})
    teachers, tagged, sentences, n_skipped = [], [], None, 0
    for offset in offsets:
        record, tagger = train_one(seed + offset, train, val, test, labels, teacher_args,
                                   device, keep_tagger=True)
        if sentences is None:      # one tokenizer for all teachers
            sentences, n_skipped = fit_sample(tagger.tokenizer, sample,
                                              tagger.model.config.max_position_embeddings)
        out = [(tuple(tags), min(marginals))
               for tags, marginals in teacher_tags(tagger, sentences, args.batch_size, device)]
        del tagger
        if str(device).startswith('cuda'):
            import torch
            torch.cuda.empty_cache()
        tagged.append(out)
        teachers.append({'seed': seed + offset, 'val_f1': record['val_f1'],
                         'test_f1': record['f1'], 'test_f1_strict': record['f1_strict'],
                         'test_macro_f1': round(F1_AVERAGES['macro'](record), 2),
                         'best_epoch': record['best_epoch'],
                         'n_silver_alone': sum(is_silver(t, [m]) for t, m in out)})
        print(f"    teacher seed {seed + offset}: val F1 {record['val_f1']:.2f}, test F1 "
              f"{record['f1']:.2f}; {teachers[-1]['n_silver_alone']} silver alone of "
              f"{len(sentences)} tagged", flush=True)
    per_sentence = list(zip(*tagged))
    silver = [(words, list(votes[0][0])) for words, votes in zip(sentences, per_sentence)
              if unanimous_silver(votes)]
    n_agreed = sum(all(t == votes[0][0] for t, _ in votes) for votes in per_sentence)
    print(f"    {len(silver)} silver of {len(sentences)} tagged sentences ({n_agreed} tagged "
          f"alike; {n_skipped} of {len(sample)} sampled skipped); students:", flush=True)
    students = {}
    for offset in offsets if student_offsets is None else student_offsets:
        student = train_one(seed + offset, train, val, test, labels, args, device,
                            test_index=test_index, silver=silver)
        student['self_training'] = {
            'n_sampled': len(sample), 'n_skipped': n_skipped, 'n_agreed': n_agreed,
            'n_silver': len(silver), 'marginal_threshold': SILVER_THRESHOLD,
            'teachers': teachers}
        students[offset] = student
    return students


# Multi-task training (--multi-task): one encoder, a linear head and a CRF per granularity
MT_HEADS = ('fine', 'coarse')


def check_aligned(fine, coarse):
    """Assert that the fine and coarse corpora list the same sentences in the same order.

    A few sentences are tokenised differently in the two files, so the check is on
    what the splits use: the number of sentences and the paper at every position.
    """
    assert len(fine) == len(coarse), \
        f'the fine and coarse corpora differ in length: {len(fine)} vs {len(coarse)} sentences'
    moved = [i for i, (f, c) in enumerate(zip(fine, coarse)) if f[2] != c[2]]
    assert not moved, f'the fine and coarse corpora differ in paper at positions {moved[:10]}'


def multitask_examples(fine, coarse):
    """Training examples of the multi-task model: [(tokens, {head: tags or None})].

    fine, coarse: the same sentences, as (tokens, tags, doc) lists of each corpus.
    A sentence with the same tokens in both is one example with both tag
    sequences. Any other gives two: (fine tokens, fine tags) and (coarse tokens,
    coarse tags), each labelled for its own head only (None for the other head).
    """
    examples = []
    for (f_tokens, f_tags, _), (c_tokens, c_tags, _) in zip(fine, coarse, strict=True):
        if f_tokens == c_tokens:
            examples.append((f_tokens, {'fine': f_tags, 'coarse': c_tags}))
        else:
            examples.append((f_tokens, {'fine': f_tags, 'coarse': None}))
            examples.append((c_tokens, {'fine': None, 'coarse': c_tags}))
    return examples


def multitask_contexts(fine, coarse, contexts):
    """The context of each example of multitask_examples(fine, coarse), in its order: the
    fine corpus's for a sentence both corpora tokenise alike, each corpus's own for the two
    examples of one they tokenise differently. contexts: {head: a (left, right) per sentence
    of that head's list (part_contexts)}."""
    out = []
    for k, ((f_tokens, _, _), (c_tokens, _, _)) in enumerate(zip(fine, coarse, strict=True)):
        out.append(contexts['fine'][k])
        if f_tokens != c_tokens:
            out.append(contexts['coarse'][k])
    return out


def encode_multitask(tokenizer, examples, label2id, max_length=512, contexts=None, budget=0):
    """encode() for every head: {head: features}, in the order of `examples`.

    An example without tags for a head has only -100 labels for it, so that
    multitask_loss never passes it to that head's CRF. contexts and budget (--context):
    as in encode(), a (left, right) per example (multitask_contexts).
    """
    features = {}
    for head in MT_HEADS:
        sentences = [(tokens, ['O'] * len(tokens) if tags[head] is None else tags[head])
                     for tokens, tags in examples]
        features[head] = encode(tokenizer, sentences, label2id[head], max_length, contexts,
                                budget)
        for feature, (_, tags) in zip(features[head], examples):
            if tags[head] is None:
                feature['labels'] = [-100] * len(feature['labels'])
    return features


def multitask_batches(features, batch_size, pad_id, shuffle_seed=None):
    """batches() of every head at once: (ids, mask, {head: labels}, idx).

    The heads' features hold the same examples in the same order, and batches()
    orders them by the shuffle seed alone, so the batches of the heads line up.
    """
    streams = [batches(features[head], batch_size, pad_id, shuffle_seed) for head in MT_HEADS]
    for parts in zip(*streams):
        ids, mask, _, idx = parts[0]
        yield ids, mask, {head: part[2] for head, part in zip(MT_HEADS, parts)}, idx


def multitask_loss(logits, labels, crfs, weights=None):
    """{head: mean CRF NLL over the rows of the batch that are labelled for the head}.

    logits {head: (B, T, L)} and labels {head: (B, T)} as from multitask_batches.
    Rows without labels for a head are dropped before its CRF, and a head with no
    labelled row in the batch is left out: it contributes nothing to the loss.
    weights (B,), optional: each row's NLL is weighted (silver rows SILVER_WEIGHT), as
    in the CRF's nll.
    """
    losses = {}
    for head, lab in labels.items():
        rows = (lab != -100).any(1)
        if rows.any():
            emissions = logits[head][rows.to(logits[head].device)]
            if weights is None:
                losses[head] = crfs[head].nll(*word_view(emissions, lab[rows]))
            else:
                losses[head] = crfs[head].nll(*word_view(emissions, lab[rows]),
                                              weights=weights[rows.to(weights.device)])
    return losses


class HeadView:
    """One head of a multi-task tagger, called like a token-classification model (predict)."""

    def __init__(self, tagger, head):
        self.tagger, self.head = tagger, head
        self.config = tagger.encoder.config         # teacher_tags reads max_position_embeddings

    def eval(self):
        self.tagger.eval()
        return self

    def __call__(self, **inputs):
        return SimpleNamespace(logits=self.tagger(**inputs)[self.head])


_TAGGER_CLASS = None


def make_multitask_tagger(model_name, label_lists):
    """Pre-trained encoder with a linear token head and a WordCRF per {head: label list}.

    Defined on first use, like the CRF, so that importing this module does not
    need torch.
    """
    global _TAGGER_CLASS
    if _TAGGER_CLASS is None:
        _TAGGER_CLASS = _define_tagger()
    return _TAGGER_CLASS(model_name, label_lists)


def _define_tagger():
    import torch
    from transformers import AutoModel

    class MultiTaskTagger(torch.nn.Module):
        """Encoder, dropout, then per head a linear layer (`heads`) and a CRF (`crfs`).

        As in a Hugging Face token-classification model: the encoder has no pooler,
        the dropout is classifier_dropout or else hidden_dropout_prob (0.1 for
        MatSciBERT), and a head starts with N(0, initializer_range) weights and zero
        bias. forward() returns {head: logits}.
        """

        def __init__(self, model_name, label_lists):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False,
                                                     dtype=torch.float32)   # as in train_one
            config = self.encoder.config
            p = getattr(config, 'classifier_dropout', None)
            self.dropout = torch.nn.Dropout(config.hidden_dropout_prob if p is None else p)
            self.heads = torch.nn.ModuleDict()
            for head, labels in label_lists.items():
                linear = torch.nn.Linear(config.hidden_size, len(labels))
                torch.nn.init.normal_(linear.weight, mean=0.0, std=config.initializer_range)
                torch.nn.init.zeros_(linear.bias)
                self.heads[head] = linear
            self.crfs = torch.nn.ModuleDict({head: make_crf(labels)
                                             for head, labels in label_lists.items()})

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = self.dropout(out.last_hidden_state)
            return {head: linear(hidden) for head, linear in self.heads.items()}

    return MultiTaskTagger


def multitask_history(log, head):
    """Per-epoch history of one head's results file.

    The single-task fields (train_loss: the summed loss of both heads; val_f1:
    this head's), then each head's own loss and validation F1.
    """
    return [{'epoch': epoch, 'train_loss': round(sum(losses.values()), 4), 'val_f1': f1[head],
             **{f'train_loss_{h}': round(losses[h], 4) for h in MT_HEADS},
             **{f'val_f1_{h}': f1[h] for h in MT_HEADS}}
            for epoch, losses, f1 in log]


def train_multitask(seed, parts, labels, args, device, test_index=None, descriptions=None,
                    silver=None, keep_tagger=False, contexts=None):
    """Fine-tune one encoder with a fine and a coarse CRF head together; test each head.

    parts: {head: (train, val, test)}, lists of (tokens, tags, doc) for the same
    corpus positions in each corpus; labels: {head: entity types}. Optimiser,
    schedule, clipping and early stopping are those of train_one with --crf;
    the training examples are those of multitask_examples, and the loss is the
    sum of the heads' mean NLLs. Each head keeps the snapshot (encoder, heads and
    CRFs) of its best validation F1 and is tested with it; training stops once
    neither head has improved for `patience` epochs. Returns {head: run record},
    each with the fields of train_one (history: see multitask_history) plus
    `multi_task`: n_examples (training examples), n_split_examples (those
    labelled for one head only, two per sentence tokenised differently) and
    other_best_epoch (the best epoch of the other head). descriptions (--label-semantics):
    {head: a text per label of labels[head]}; each head's linear layer is then replaced
    by a label-semantic head over their embeddings by the pre-trained encoder, as in
    train_one. silver (multi-task self-training): [(words, {head: tags or None})] trained
    on with the gold examples, each row's NLL weighted SILVER_WEIGHT (a head without silver
    tags for a sentence does not see it). keep_tagger: return (records, tagger), the
    tagger holding the model, the tokenizer, label2id and id2label per head, and the
    snapshot of each head's best epoch (`snapshots`; mt_teacher_tags). contexts (--context):
    {head: {'train'|'val'|'test': part_contexts of that part of the head's corpus}}; each
    sentence is encoded with up to args.context subwords of context per side, as in train_one
    (a training example takes its context as in multitask_contexts).
    """
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed

    set_seed(seed)
    label_lists = {h: ['O'] + [f'{p}-{t}' for t in labels[h] for p in ('B', 'I')]
                   for h in MT_HEADS}
    label2id = {h: {lab: i for i, lab in enumerate(lst)} for h, lst in label_lists.items()}
    id2label = {h: dict(enumerate(lst)) for h, lst in label_lists.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    assert tokenizer.is_fast, 'a fast tokenizer is required for word_ids()'
    model = make_multitask_tagger(args.model, label_lists).to(device)
    if descriptions is not None:
        # embedded before any fine-tuning, head by head; each replaces its linear layer
        for h in MT_HEADS:
            assert len(descriptions[h]) == len(labels[h]), 'one description per label'
            embeddings = embed_descriptions(model.encoder, tokenizer, descriptions[h], device)
            model.heads[h] = make_label_head(embeddings, model.encoder.config).to(device)
    pad_id = tokenizer.pad_token_id

    examples = multitask_examples(parts['fine'][0], parts['coarse'][0])
    if contexts is None:
        f_train = encode_multitask(tokenizer, examples, label2id)
    else:
        assert not silver, 'silver sentences have no document context'
        f_train = encode_multitask(
            tokenizer, examples, label2id, model.encoder.config.max_position_embeddings,
            multitask_contexts(parts['fine'][0], parts['coarse'][0],
                               {h: contexts[h]['train'] for h in MT_HEADS}), args.context)
    weight = None
    if silver:
        # the gold examples, then the silver ones; multitask_batches shuffles them together
        extra = encode_multitask(tokenizer, silver, label2id,
                                 model.encoder.config.max_position_embeddings)
        for h in MT_HEADS:
            f_train[h] += extra[h]
        weight = [1.0] * len(examples) + [SILVER_WEIGHT] * len(silver)
    if contexts is None:
        f_val, f_test = ({h: encode(tokenizer, [(t, g) for t, g, _ in parts[h][k]], label2id[h])
                          for h in MT_HEADS} for k in (1, 2))
    else:
        f_val, f_test = ({h: encode(tokenizer, [(t, g) for t, g, _ in parts[h][k]], label2id[h],
                                    model.encoder.config.max_position_embeddings,
                                    contexts[h][name], args.context)
                          for h in MT_HEADS} for k, name in ((1, 'val'), (2, 'test')))

    optimizer = torch.optim.AdamW(
        [{'params': [*model.encoder.parameters(), *model.heads.parameters()]},
         {'params': model.crfs.parameters(), 'lr': CRF_LR, 'weight_decay': 0.0}],
        lr=args.lr, weight_decay=0.01)
    params = list(model.parameters())
    steps = args.epochs * ((len(f_train['fine']) + args.batch_size - 1) // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)

    def val_scores():
        f1 = {}
        for h in MT_HEADS:
            pred = predict(HeadView(model, h), f_val[h], args.batch_size, pad_id, id2label[h],
                           device, model.crfs[h])
            f1[h] = score([g for _, g, _ in parts[h][1]], pred)['f1']
        return f1

    best = dict.fromkeys(MT_HEADS, (-1.0, None, -1))     # per head: (val F1, snapshot, epoch)
    log = []
    for epoch in range(args.epochs):
        model.train()
        totals = dict.fromkeys(MT_HEADS, 0.0)
        for ids, mask, lab, idx in multitask_batches(f_train, args.batch_size, pad_id,
                                                      shuffle_seed=seed * 1000 + epoch):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device))
            losses = multitask_loss(logits, lab, model.crfs, weights=None if weight is None else
                                    torch.tensor([weight[i] for i in idx], device=device))
            loss = sum(losses.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            for h, head_loss in losses.items():
                totals[h] += head_loss.item()
        val_f1 = val_scores()
        log.append((epoch, totals, val_f1))
        print(f"    seed {seed} epoch {epoch}: loss fine {totals['fine']:.3f} coarse "
              f"{totals['coarse']:.3f}, val F1 fine {val_f1['fine']:.2f} coarse "
              f"{val_f1['coarse']:.2f}", flush=True)
        improved = [h for h in MT_HEADS if val_f1[h] > best[h][0]]
        if improved:
            snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            for h in improved:
                best[h] = (val_f1[h], snapshot, epoch)
        elif epoch - max(b[2] for b in best.values()) >= args.patience:
            break

    n_split = sum(1 for _, tags in examples if None in tags.values())
    records = {}
    biases = entity_biases(args)
    for h in MT_HEADS:
        model.load_state_dict(best[h][1])
        train, val, test = parts[h]
        if biases:
            decodes = biased_decodes(HeadView(model, h), f_test[h], args.batch_size, pad_id,
                                     id2label[h], device, model.crfs[h], biases)
            test_pred = decodes[0.0]
        else:
            test_pred = predict(HeadView(model, h), f_test[h], args.batch_size, pad_id,
                                id2label[h], device, model.crfs[h])
        result = score([g for _, g, _ in test], test_pred)
        other = [o for o in MT_HEADS if o != h][0]
        result.update({'seed': seed, 'best_epoch': best[h][2], 'val_f1': best[h][0],
                       'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
                       'history': multitask_history(log, h),
                       'multi_task': {'n_examples': len(examples), 'n_split_examples': n_split,
                                      'other_best_epoch': best[other][2]}})
        if getattr(args, 'save_predictions', False):
            result.update(prediction_fields(test_index, test, test_pred))
        if biases:
            result.update(bias_fields(
                biases, test, decodes, val,
                biased_decodes(HeadView(model, h), f_val[h], args.batch_size, pad_id,
                               id2label[h], device, model.crfs[h], biases),
                getattr(args, 'save_predictions', False)))
        records[h] = result
    if keep_tagger:
        return records, SimpleNamespace(model=model, tokenizer=tokenizer, label2id=label2id,
                                        id2label=id2label,
                                        snapshots={h: best[h][1] for h in MT_HEADS})
    return records


# Multi-task self-training (--multi-task --self-train --teachers N; P1 "Next, after week 4"
# item 2): silver filters (teachers that agree, least marginal), the strictest first; the
# first is the week-4 rule (unanimous, 0.9)
SILVER_FILTERS = ((3, 0.9), (3, 0.8), (3, 0.7), (2, 0.9), (2, 0.8))


def filtered_tags(votes, agree, threshold):
    """A sentence's silver tags under a filter, or None.

    votes: [(tags, least marginal)] of the teachers. The tags must be given alike by at
    least `agree` teachers, each of them with every marginal at least `threshold`, and
    have an entity (with 3 teachers, (3, SILVER_THRESHOLD) is unanimous_silver).
    """
    tags, n = Counter(t for t, _ in votes).most_common(1)[0]
    if n < agree or not any(t.startswith('B-') for t in tags):
        return None
    if min(least for t, least in votes if t == tags) < threshold:
        return None
    return list(tags)


def choose_filter(val_votes, gold):
    """((agree, threshold), table): the silver filter of a head, chosen on the validation
    sentences.

    val_votes: the teachers' votes on each validation sentence; gold: its tags. The table
    has, per filter of SILVER_FILTERS, the sentences it selects (n) and how many of them
    have the gold tags (right). The chosen filter selects the most sentences among those
    right at least as often as the strictest; the strictest if that selects none.
    """
    table = []
    for agree, threshold in SILVER_FILTERS:
        chosen = [(filtered_tags(v, agree, threshold), list(g)) for v, g in zip(val_votes, gold)]
        chosen = [(t, g) for t, g in chosen if t is not None]
        table.append({'agree': agree, 'threshold': threshold, 'n': len(chosen),
                      'right': sum(t == g for t, g in chosen)})
    if not table[0]['n']:
        return SILVER_FILTERS[0], table
    precision = table[0]['right'] / table[0]['n']
    ok = [row for row in table if row['n'] and row['right'] / row['n'] >= precision]
    best = max(ok, key=lambda row: row['n'])        # ties: the stricter, which comes first
    return (best['agree'], best['threshold']), table


def mt_teacher_tags(tagger, sentences, batch_size, device):
    """{head: teacher_tags of sentences[head]} by a multi-task tagger (train_multitask with
    keep_tagger), each head with the snapshot of its best epoch."""
    out = {}
    for h in MT_HEADS:
        tagger.model.load_state_dict(tagger.snapshots[h])
        view = SimpleNamespace(model=HeadView(tagger.model, h), crf=tagger.model.crfs[h],
                               tokenizer=tagger.tokenizer, label2id=tagger.label2id[h],
                               id2label=tagger.id2label[h])
        out[h] = teacher_tags(view, sentences[h], batch_size, device)
    return out


def train_mt_with_teachers(seed, offsets, parts, labels, args, device, sample, test_index=None,
                           student_offsets=None):
    """One multi-task self-training run: {offset: {head: the record of that student}}.

    The teachers are train_multitask on the run's data with the model seeds seed + offset.
    Each tags the sample with both heads, and the validation sentences of each head. A
    head's silver filter is chosen on its validation sentences (choose_filter), and a
    sentence is silver for a head if the filter keeps it (filtered_tags). Each seed of
    `student_offsets` (default: the teachers') then trains a multi-task student on the gold
    data plus the silver sentences, each labelled for the heads it is silver for. Every
    record adds `self_training`: n_sampled, n_skipped, n_silver per head, n_silver_sentences,
    the filters (with their validation tables) and the teachers' scores per head.
    """
    teacher_args = SimpleNamespace(**{**vars(args), 'save_predictions': False})
    val_words = {h: [t for t, _, _ in parts[h][1]] for h in MT_HEADS}
    teachers, tagged, val_tagged, sentences, n_skipped = [], [], [], None, 0
    for offset in offsets:
        records, tagger = train_multitask(seed + offset, parts, labels, teacher_args, device,
                                          keep_tagger=True)
        if sentences is None:      # one tokenizer for all teachers
            sentences, n_skipped = fit_sample(tagger.tokenizer, sample,
                                              tagger.model.encoder.config.max_position_embeddings)
        for store, words in ((tagged, {h: sentences for h in MT_HEADS}), (val_tagged, val_words)):
            out = mt_teacher_tags(tagger, words, args.batch_size, device)
            store.append({h: [(tuple(t), min(m)) for t, m in out[h]] for h in MT_HEADS})
        del tagger
        if str(device).startswith('cuda'):
            import torch
            torch.cuda.empty_cache()
        teachers.append({'seed': seed + offset, **{h: {
            'val_f1': records[h]['val_f1'], 'test_f1': records[h]['f1'],
            'test_f1_strict': records[h]['f1_strict'],
            'test_macro_f1': round(F1_AVERAGES['macro'](records[h]), 2),
            'best_epoch': records[h]['best_epoch']} for h in MT_HEADS}})
        print(f"    teacher seed {seed + offset}: test F1 fine {records['fine']['f1']:.2f}, "
              f"coarse {records['coarse']['f1']:.2f}", flush=True)
    filters, silver_tags = {}, {}
    for h in MT_HEADS:
        votes = list(zip(*(t[h] for t in val_tagged)))
        filters[h] = choose_filter(votes, [g for _, g, _ in parts[h][1]])
        silver_tags[h] = [filtered_tags(v, *filters[h][0])
                          for v in zip(*(t[h] for t in tagged))]
    silver = [(words, {h: silver_tags[h][i] for h in MT_HEADS})
              for i, words in enumerate(sentences)
              if any(silver_tags[h][i] is not None for h in MT_HEADS)]
    info = {'n_sampled': len(sample), 'n_skipped': n_skipped,
            'n_silver': {h: sum(t is not None for t in silver_tags[h]) for h in MT_HEADS},
            'n_silver_sentences': len(silver),
            'filters': {h: {'agree': filters[h][0][0], 'threshold': filters[h][0][1],
                            'validation': filters[h][1]} for h in MT_HEADS},
            'teachers': teachers}
    print(f"    silver: {info['n_silver']} of {len(sentences)} tagged sentences (filters "
          f"{ {h: filters[h][0] for h in MT_HEADS} }); students:", flush=True)
    students = {}
    for offset in offsets if student_offsets is None else student_offsets:
        records = train_multitask(seed + offset, parts, labels, args, device,
                                  test_index=test_index, silver=silver)
        for h in MT_HEADS:
            records[h]['self_training'] = info
        students[offset] = records
    return students


# F1 averages reported for every run (seqeval, in %): micro is the primary one
F1_AVERAGES = {
    'micro': lambda r: r['f1'],
    'micro_strict': lambda r: r['f1_strict'],
    'macro': lambda r: 100 * r['per_class']['macro avg']['f1-score'],
    'weighted': lambda r: 100 * r['per_class']['weighted avg']['f1-score'],
}

# Results files: ner_results_<granularity>[_<protocol>][_<variant>].json. The protocol
# part is absent for the fixed split, the variant part for the baseline. The protocol
# is tried first, so the baseline file of a protocol never reads as a variant named
# like the protocol.
RESULT_FILE = re.compile(r'ner_results_(fine|coarse|sofc_slot|sofc)'
                         r'(?:_(random_split_vary|leave_one_paper_out|leave_two_papers_out|'
                         r'published_split))?'
                         r'(?:_([a-z0-9_]+))?\.json')
# untagged variants: a tagged variant must not take the name of one (--crf --tag mt: crf_mt)
PLAIN_VARIANTS = ('baseline', 'cw', 'crf', 'crf_mt', 'crf_ls', 'crf_st', 'crf_mt_ls', 'crf_st3',
                  'crf_ctx', 'crf_mt_st3', 'crf_mt_ctx', 'crf_mr', 'crf_bias', 'crf_mt_bias',
                  'crf_ls_bias', 'crf_mt_ls_bias', 'crf_ctx_bias', 'crf_mt_ctx_bias',
                  'crf_mr_bias', 'crf_eb', 'crf_ebv')


def variant_name(args):
    """Variant of a run: 'baseline', 'cw', 'crf', 'crf_mt', 'crf_ls', 'crf_mt_ls', 'crf_st' or
    'crf_st<N>', 'crf_mt_st<N>', 'crf_ctx', 'crf_mt_ctx' or 'crf_mr', with _bias for
    --entity-bias (crf_bias, crf_ctx_bias), then the tag.

    'cw' is --class-weights, 'crf_mt' is --crf --multi-task, 'crf_ls' is --crf
    --label-semantics, 'crf_mt_ls' both, 'crf_st' is --crf --self-train and 'crf_st<N>' that
    with --teachers N ('crf_mt_st<N>' with --multi-task too). A tag is appended with an underscore (crf_m100); the baseline is
    named by it alone.
    """
    if getattr(args, 'crf', False):
        multi_task = getattr(args, 'multi_task', False)
        name = 'crf_mt' if multi_task else 'crf'
        if getattr(args, 'label_semantics', False):
            name = 'crf_mt_ls' if multi_task else 'crf_ls'
        if getattr(args, 'self_train', None) is not None:
            teachers = getattr(args, 'teachers', 1)
            name = 'crf_st' if teachers == 1 else f'crf_st{teachers}'
            if multi_task:
                name = f'crf_mt_st{teachers}'
        if getattr(args, 'context', 0):
            name = 'crf_mt_ctx' if multi_task else 'crf_ctx'
        if getattr(args, 'augment_mentions', 0):
            name = 'crf_mr'
        if hasattr(args, 'entity_bias'):
            name += '_bias'
    else:
        name = 'cw' if args.class_weights else 'baseline'
    tag = getattr(args, 'tag', None)
    if not tag:
        return name
    return tag if name == 'baseline' else f'{name}_{tag}'


def reference_variant(variant):
    """Variant that a variant is compared with ('baseline': the baseline file).

    crf_mt[_<tag>], crf_ls[_<tag>], crf_mt_ls[_<tag>], crf_st[N][_<tag>] and crf_<tag> are
    compared with crf and cw_<tag> with cw; cw, crf and a bare tag with the baseline.
    """
    for base in ('crf', 'cw'):
        if variant.startswith(f'{base}_'):
            return base
    return 'baseline'


def result_suffix(protocol, variant):
    """File-name part after the granularity (none for the fixed split and the baseline)."""
    suffix = '' if protocol == 'random_split_fixed' else f'_{protocol}'
    return suffix if variant == 'baseline' else f'{suffix}_{variant}'


def parse_result_name(fn):
    """(granularity, protocol, variant) of a results file name; None for other files.

    The protocol is as in the results ('random_split_fixed' without a protocol
    part) and the variant is 'baseline' for the baseline file of a protocol.
    """
    m = RESULT_FILE.fullmatch(fn)
    if not m:
        return None
    return m.group(1), m.group(2) or 'random_split_fixed', m.group(3) or 'baseline'


def stored_settings(results):
    """Settings of a results file that its runs must share with a run that resumes them.

    Files written before variants existed hold baseline runs, files without
    a stored offset have offset 0, and only --self-train files store a corpus
    SHA-256 (None otherwise).
    """
    return {'labels': results.get('labels'), 'model': results.get('model'),
            'variant': results.get('variant', 'baseline'),
            'seed_offset': results.get('config', {}).get('seed_offset', 0),
            'self_train_sha256': results.get('config', {}).get('self_train_sha256')}


def resumable_runs(path, labels, model, variant, seed_offset=0, save_predictions=False,
                   self_train_sha256=None):
    """Runs of an earlier invocation that this one may reuse.

    Only runs with the same labels, model, training variant, seed offset and
    self-training corpus count (see stored_settings). With `save_predictions`,
    runs that did not store their predictions are not reused.
    """
    if not os.path.exists(path):
        return []
    with open(path) as f:
        previous = json.load(f)
    same = stored_settings(previous) == {'labels': labels, 'model': model, 'variant': variant,
                                         'seed_offset': seed_offset,
                                         'self_train_sha256': self_train_sha256}
    runs = previous.get('runs', []) if same else []
    return [r for r in runs if 'test_pred' in r] if save_predictions else runs


class OverwriteError(ValueError):
    """A results file holds runs that a new run could not resume and would replace."""


def setting_change(name, stored, now):
    """How a stored setting (see stored_settings) differs from that of this invocation."""
    if name == 'labels' and isinstance(stored, list):
        only_file, only_now = sorted(set(stored) - set(now)), sorted(set(now) - set(stored))
        if not (only_file or only_now):
            return 'labels: the same classes in another order'
        return f'labels: {only_file} only in the file, {only_now} only now'
    return f'{name}: {stored!r} in the file, {now!r} now'


def check_overwrite(path, labels, model, variant, seed_offset=0, save_predictions=False,
                    self_train_sha256=None):
    """Raise OverwriteError if `path` holds runs that resumable_runs would not reuse.

    A run writes its results file anew, so those runs would be lost: runs made
    with other labels, model, variant, seed offset or self-training corpus, and,
    with `save_predictions`, runs that did not store their predictions.
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        previous = json.load(f)
    held = previous.get('runs', [])
    kept = resumable_runs(path, labels, model, variant, seed_offset, save_predictions,
                          self_train_sha256)
    if len(kept) >= len(held):
        return
    stored = stored_settings(previous)
    now = {'labels': labels, 'model': model, 'variant': variant, 'seed_offset': seed_offset,
           'self_train_sha256': self_train_sha256}
    reasons = [setting_change(k, stored[k], now[k]) for k in stored if stored[k] != now[k]]
    if not reasons:
        missing = [r.get('run_id') for r in held if 'test_pred' not in r]
        reasons = [f'--save-predictions is set and runs {missing} have no stored predictions']
    raise OverwriteError(
        f"{path}: runs that this invocation cannot resume ({'; '.join(reasons)}) would be "
        f"replaced ({len(held) - len(kept)} of {len(held)} stored). Use --tag NAME to write "
        f"the new runs to a file of their own, or --overwrite to replace the old runs on purpose")


def check_results_files(args, files, self_train_sha256=None):
    """check_overwrite for every (path, labels) that a run will write.

    Nothing is checked with --overwrite, or for smoke-test files (--max-train),
    which may be replaced at will.
    """
    if args.max_train or getattr(args, 'overwrite', False):
        return
    for path, labels in files:
        check_overwrite(path, labels, args.model, variant_name(args), args.seed_offset,
                        args.save_predictions, self_train_sha256)


def summarize(results):
    """Aggregate the runs of one protocol into the results dict (in place).

    test_f1_mean / test_f1_std / test_f1_strict_mean keep their meaning
    (micro F1, population SD). `summary` adds, per F1 average, the mean, the
    population SD, the 95% Student-t CI of the mean and the per-run values.
    """
    runs = results['runs']
    if not runs:
        return results
    f1s = [r['f1'] for r in runs]
    results['test_f1_mean'] = round(statistics.mean(f1s), 2)
    results['test_f1_std'] = round(statistics.pstdev(f1s), 2) if len(f1s) > 1 else 0.0
    results['test_f1_strict_mean'] = round(statistics.mean(r['f1_strict'] for r in runs), 2)
    series = {name: [get(r) for r in runs] for name, get in F1_AVERAGES.items()}
    summary = {'n_runs': len(runs), 'run_ids': [r.get('run_id') for r in runs]}
    for name, vals in series.items():
        summary[name] = {
            'mean': round(statistics.mean(vals), 2),
            'pop_sd': round(statistics.pstdev(vals), 2) if len(vals) > 1 else 0.0,
            'ci95': [round(x, 2) for x in metrics.mean_ci(vals)['ci']],
            'per_run': [round(v, 2) for v in vals],
        }
    results['summary'] = summary
    return results


def compare_to_baseline(base, variant):
    """Paired differences (variant - baseline) over runs with the same run_id.

    Runs are paired by run_id (seed<k> for random splits, the held-out paper
    for leave-one-paper-out), so both members of a pair share the split (and
    the model seed, unless one file used --seed-offset). `base` is the
    reference results (the baseline, or crf / cw for tagged variants of those).
    For every F1 average: the per-run differences, their mean and the 95%
    Student-t CI of the mean difference.
    """
    base_runs = {r.get('run_id'): r for r in base['runs']}
    pairs = [(base_runs[r['run_id']], r) for r in variant['runs']
             if r.get('run_id') in base_runs]
    out = {'n_pairs': len(pairs), 'run_ids': [v['run_id'] for _, v in pairs]}
    for name, get in F1_AVERAGES.items():
        diffs = [get(v) - get(b) for b, v in pairs]
        out[name] = {
            'mean_diff': round(statistics.mean(diffs), 2) if diffs else 0.0,
            'ci95': [round(x, 2) for x in metrics.mean_ci(diffs)['ci']],
            'per_run': [round(d, 2) for d in diffs],
        }
    return out


def compare_all(directory):
    """Pair every variant results file in `directory` with its reference file.

    The reference (see reference_variant) has the same granularity and
    protocol; smoke-test files and the baseline files themselves are not
    compared. Writes <directory>/ner_comparisons.json and returns its content.
    """
    comparisons = {}
    for fn in sorted(os.listdir(directory)):
        parsed = parse_result_name(fn)
        if parsed is None or parsed[2] == 'baseline' or 'smoke' in parsed[2]:
            continue
        granularity, protocol, variant = parsed
        reference = reference_variant(variant)
        ref_fn = f"ner_results_{granularity}{result_suffix(protocol, reference)}.json"
        if not os.path.exists(os.path.join(directory, ref_fn)):
            print(f"  {fn}: no reference file {ref_fn}, skipped")
            continue
        with open(os.path.join(directory, ref_fn)) as f:
            ref = json.load(f)
        with open(os.path.join(directory, fn)) as f:
            var = json.load(f)
        comp = compare_to_baseline(ref, var)
        comparisons[fn[:-len('.json')]] = {
            'variant': variant, 'granularity': granularity,
            'protocol': var.get('protocol'), 'reference': reference,
            'reference_file': ref_fn, **comp}
        d = comp['micro']
        print(f"  {fn}: micro F1 difference {d['mean_diff']:+.2f} vs {reference}, "
              f"95% CI {d['ci95']} over {comp['n_pairs']} paired runs")
    with open(os.path.join(directory, 'ner_comparisons.json'), 'w') as f:
        json.dump(comparisons, f, indent=2)
    return comparisons


def paper_fold(data, held_out, split_seed, indices=False):
    """(train, val, test) when the papers `held_out` are the test set: the other papers'
    sentences, shuffled with random.Random(split_seed), 15% of them for validation."""
    rest = [i for i, x in enumerate(data) if x[2] not in held_out]
    order = list(range(len(rest)))
    random.Random(split_seed).shuffle(order)
    n_val = round(0.15 * len(rest))
    val = [rest[i] for i in order[:n_val]]
    train = [rest[i] for i in order[n_val:]]
    test = [i for i, x in enumerate(data) if x[2] in held_out]
    if not indices:
        train, val, test = ([data[i] for i in part] for part in (train, val, test))
    return train, val, test


def make_folds(data, args, indices=False, granularity=None):
    """[(name, train, val, test)] for leave-one-paper-out, leave-two-papers-out (a fold per
    pair of papers, named <paper>+<paper>) and the published split (one fold per model seed,
    named seed<K>, all with the partition of `granularity`); None for random splits.

    The parts are lists of data tuples, or of their positions in `data` when
    `indices` is set (the test positions go with the stored predictions).
    """
    if getattr(args, 'published_split', False):
        parts, _ = published_split(data, granularity)
        if not indices:
            parts = [[data[i] for i in part] for part in parts]
        return [(f'seed{seed}', *parts) for seed in args.seeds]
    if args.leave_one_paper_out:
        docs = sorted({d for _, _, d in data})
        keep = held_out_papers(docs, getattr(args, 'papers', None))
        return [(doc, *paper_fold(data, (doc,), args.split_seed, indices))
                for doc in docs if doc in keep]
    if getattr(args, 'leave_two_papers_out', False):
        docs = sorted({d for _, _, d in data})
        return [('+'.join(pair), *paper_fold(data, pair, args.split_seed, indices))
                for pair in itertools.combinations(docs, 2)]
    return None


def held_out_papers(docs, papers):
    """The folds to run (--papers): all of `docs`, or those named, with or without .conll.

    Every fold is built as before and only the others are left out, so a fold run on its
    own is the same as within the full protocol.
    """
    if not papers:
        return set(docs)
    stems = {doc.removesuffix('.conll'): doc for doc in docs}
    unknown = [p for p in papers if p.removesuffix('.conll') not in stems]
    if unknown:
        raise SystemExit(f"--papers: no held-out paper {', '.join(unknown)}; "
                         f"the papers are {', '.join(sorted(stems))}")
    return {stems[p.removesuffix('.conll')] for p in papers}


def granularity_data(granularity, args):
    """(data, labels, tie info) of a granularity, tags of the classes not kept mapped to O.

    Fine: the --top-k classes; SOFC-Exp's slots: SOFC_SLOT_TYPES; otherwise all types.
    """
    data = load_split_corpus(granularity)
    labels, tie_info = select_labels(data, args.top_k if granularity == 'fine' else None)
    if granularity == 'sofc_slot':
        labels = [t for t in labels if t in SOFC_SLOT_TYPES]
    keep = set(labels)
    data = [(t, restrict_tags(g, keep), d) for t, g, d in data]
    if getattr(args, 'published_split', False):
        _, data = published_split(data, granularity)      # the release's tags
    return data, labels, tie_info


def granularities_of(args):
    """The granularities that --granularity names: one, or a dataset's two levels."""
    return list(GRANULARITY_PAIRS.get(args.granularity, (args.granularity,)))


def head_granularities(args):
    """{multi-task head: its granularity}: MaterioMiner's fine and coarse levels, or SOFC-Exp's
    slots ('fine' head) and entity types ('coarse' head)."""
    return dict(zip(MT_HEADS, GRANULARITY_PAIRS[args.granularity]))


def protocol_name(args):
    if getattr(args, 'published_split', False):
        return PUBLISHED_SPLIT
    if getattr(args, 'leave_two_papers_out', False):
        return 'leave_two_papers_out'
    return 'leave_one_paper_out' if args.leave_one_paper_out else (
        'random_split_vary' if args.vary_split else 'random_split_fixed')


def held_out_protocol(args):
    """Whether the test papers are unseen (leave-one- or leave-two-papers-out)."""
    return args.leave_one_paper_out or getattr(args, 'leave_two_papers_out', False)


def results_path(out_dir, granularity, args):
    """<out_dir>/ner_results_<granularity>[_<protocol>][_<variant>][_smoke<N>].json"""
    suffix = result_suffix(protocol_name(args), variant_name(args))
    if args.max_train:
        suffix += f'_smoke{args.max_train}'
    return os.path.join(out_dir, f'ner_results_{granularity}{suffix}.json')


def stored_config(args):
    """The arguments saved as `config` (the granularity is a field of its own)."""
    return {k: v for k, v in vars(args).items()
            if k not in ('granularity', 'compare', 'overwrite', 'papers')}


def torch_device(args):
    """Set the number of torch threads and return the device to train on.

    A CUDA device is made deterministic first (cuda_determinism).
    """
    import torch
    torch.set_num_threads(args.threads or os.cpu_count())
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device.startswith('cuda'):
        cuda_determinism()
    return device


def cuda_determinism():
    """Deterministic CUDA training (ROADMAP §4, week-4 protocol).

    PyTorch's deterministic algorithms (an operation without one raises an error),
    deterministic cuDNN without autotuning, the fixed cuBLAS workspace that deterministic
    cuBLAS needs (read when the first CUDA handle is made, so set before any CUDA work),
    and attention by the math kernel of scaled_dot_product_attention: the fused kernels
    do not have a deterministic backward on every GPU. Only CUDA runs call it, so CPU
    runs are unchanged.
    """
    import torch
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for fused in ('enable_flash_sdp', 'enable_mem_efficient_sdp', 'enable_cudnn_sdp'):
        if hasattr(torch.backends.cuda, fused):
            getattr(torch.backends.cuda, fused)(False)


def run_environment(device):
    """Where a CUDA run trained, stored with every GPU run as `environment`; None on a CPU.

    The GPU, the CUDA, cuDNN, torch and transformers versions, and the determinism settings.
    """
    if not device.startswith('cuda'):
        return None
    import torch
    import transformers
    return {'device': torch.cuda.get_device_name(torch.device(device)),
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def run(args):
    out_dir = os.path.join(PROCESSED_DIR, 'ner')
    if getattr(args, 'multi_task', False):
        if getattr(args, 'self_train', None) is not None:
            run_mt_teachers(args, out_dir)
        else:
            run_multitask(args, out_dir)
        return
    if hasattr(args, 'teachers'):
        run_teachers(args, out_dir)
        return
    granularities = granularities_of(args)
    tasks = {g: granularity_data(g, args) for g in granularities}
    semantics = getattr(args, 'label_semantics', False)
    descriptions = {g: label_descriptions(g, tasks[g][1]) if semantics else None
                    for g in granularities}
    self_train = getattr(args, 'self_train', None) is not None
    # --self-train: the sample, and the corpus path, SHA-256, sample size and silver weight
    sample, corpus = (self_training_corpus(args.self_train, getattr(args, 'self_train_sample',
                                                                    SELF_TRAIN_SAMPLE))
                      if self_train else (None, {}))
    # every results file is checked before any training, so a run never stops halfway
    check_results_files(args, [(results_path(out_dir, g, args), tasks[g][1])
                               for g in granularities], corpus.get('self_train_sha256'))
    device = torch_device(args)
    environment = run_environment(device)
    os.makedirs(out_dir, exist_ok=True)

    for granularity in granularities:
        data, labels, tie_info = tasks[granularity]
        protocol = protocol_name(args)
        variant = variant_name(args)
        out_path = results_path(out_dir, granularity, args)
        results = {'granularity': granularity, 'protocol': protocol, 'variant': variant,
                   'model': args.model, 'labels': labels, 'label_tie': tie_info,
                   'paper_test_f1': PAPER_F1[granularity], 'config': stored_config(args),
                   'runs': resumable_runs(out_path, labels, args.model, variant,
                                          args.seed_offset, args.save_predictions,
                                          corpus.get('self_train_sha256'))}
        if semantics:
            results['config']['label_descriptions_sha256'] = ner_label_text.descriptions_sha256(
                labels, descriptions[granularity])
        if self_train:
            results['config'].update(corpus)
        done = {r.get('run_id') for r in results['runs']}

        offset = f", seed offset {args.seed_offset}" if args.seed_offset else ''
        print(f"\n=== NER {granularity}: {len(data)} sentences, {len(labels)} classes, "
              f"protocol {protocol}, variant {variant}{offset}, device {device} ===")
        if tie_info:
            print(f"  label cut tie: {tie_info}")
        if semantics:
            print(f"  label descriptions: sha256 "
                  f"{results['config']['label_descriptions_sha256'][:12]}")
        if self_train:
            print(f"  self-training: {len(sample)} sentences sampled from {corpus['self_train']} "
                  f"(sha256 {corpus['self_train_sha256'][:12]})")

        folds = make_folds(data, args, indices=True,     # parts as positions in `data`
                           granularity=granularity)
        if folds is None:
            folds = []
            for seed in args.seeds:
                split_seed = seed if args.vary_split else args.split_seed
                folds.append((f'seed{seed}', *random_split(len(data), split_seed)))
        for run_id, tr, va, te in folds:
            if run_id in done:
                print(f"  {run_id}: already done, skipping")
                continue
            if args.max_train:
                tr = tr[:args.max_train]
            train, val, test = ([data[i] for i in part] for part in (tr, va, te))
            seed = int(run_id[4:]) if run_id.startswith('seed') else args.seeds[0]
            seed += args.seed_offset      # the model seed moves, the split and run_id do not
            start = time.time()
            if self_train:
                res = train_self_training(seed, train, val, test, labels, args, device, sample,
                                          test_index=te)
            else:
                texts = load_full_texts(granularity) if getattr(args, 'context', 0) else None
                contexts = ({name: part_contexts(data, part, texts)
                             for name, part in (('train', tr), ('val', va), ('test', te))}
                            if getattr(args, 'context', 0) else None)
                augmented = (mention_replacements([(w, t) for w, t, _ in train],
                                                  args.augment_mentions, seed)
                             if getattr(args, 'augment_mentions', 0) else None)
                res = train_one(seed, train, val, test, labels, args, device, test_index=te,
                                descriptions=descriptions[granularity], contexts=contexts,
                                augmented=augmented)
                if augmented is not None:
                    res['n_augmented'] = len(augmented)
            res['run_id'] = run_id
            res['runtime_sec'] = round(time.time() - start, 1)
            if environment:
                res['environment'] = environment
            results['runs'].append(res)
            summarize(results)
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)
            # the published numbers are in-distribution; no reference for held-out papers
            ref = ('' if held_out_protocol(args) or PAPER_F1[granularity] is None
                   else f" (paper {PAPER_F1[granularity]})")
            print(f"  {run_id}: test F1 {res['f1']:.2f} (strict {res['f1_strict']:.2f}), "
                  f"best epoch {res['best_epoch']}, {res['runtime_sec']:.0f}s; "
                  f"running mean {results['test_f1_mean']:.2f}{ref}", flush=True)
        if results['runs']:
            summarize(results)
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)
            s = results['summary']
            print(f"  micro F1 {s['micro']['mean']:.2f} 95% CI {s['micro']['ci95']}, "
                  f"macro {s['macro']['mean']:.2f}, weighted {s['weighted']['mean']:.2f} "
                  f"over {s['n_runs']} runs")
        print(f"Saved {out_path}")


def run_multitask(args, out_dir):
    """run() for --multi-task: one model per run_id, a results file per granularity.

    Labels, splits (by corpus position), run ids and model seeds are those of the
    single-task runs. Both files are checked (check_results_files) before
    training and written after every run; a run_id counts as done only if both
    files hold it.
    """
    gran = head_granularities(args)
    data, labels, ties = {}, {}, {}
    for head in MT_HEADS:
        data[head], labels[head], ties[head] = granularity_data(gran[head], args)
    check_aligned(data['fine'], data['coarse'])
    semantics = getattr(args, 'label_semantics', False)
    descriptions = ({head: label_descriptions(head, labels[head]) for head in MT_HEADS}
                    if semantics else None)
    paths = {head: results_path(out_dir, gran[head], args) for head in MT_HEADS}
    check_results_files(args, [(paths[head], labels[head]) for head in MT_HEADS])
    device = torch_device(args)
    environment = run_environment(device)
    os.makedirs(out_dir, exist_ok=True)
    protocol, variant, config = protocol_name(args), variant_name(args), stored_config(args)
    results = {head: {'granularity': gran[head], 'protocol': protocol, 'variant': variant,
                      'model': args.model, 'labels': labels[head], 'label_tie': ties[head],
                      'paper_test_f1': PAPER_F1[gran[head]], 'config': dict(config),
                      'runs': resumable_runs(paths[head], labels[head], args.model, variant,
                                             args.seed_offset, args.save_predictions)}
               for head in MT_HEADS}
    if semantics:
        for head in MT_HEADS:
            results[head]['config']['label_descriptions_sha256'] = \
                ner_label_text.descriptions_sha256(labels[head], descriptions[head])
    done = set.intersection(*({r.get('run_id') for r in results[h]['runs']} for h in MT_HEADS))

    def save():
        for head in MT_HEADS:
            summarize(results[head])
            with open(paths[head], 'w') as f:
                json.dump(results[head], f, indent=2)

    offset = f", seed offset {args.seed_offset}" if args.seed_offset else ''
    print(f"\n=== NER multi-task: {len(data['fine'])} sentences, {len(labels['fine'])} "
          f"{gran['fine']} and {len(labels['coarse'])} {gran['coarse']} classes, protocol "
          f"{protocol}, variant {variant}"
          f"{offset}, device {device} ===")
    for head in MT_HEADS:
        if ties[head]:
            print(f"  {head} label cut tie: {ties[head]}")
        if semantics:
            print(f"  {head} label descriptions: sha256 "
                  f"{results[head]['config']['label_descriptions_sha256'][:12]}")

    folds = make_folds(data['fine'], args, indices=True,     # parts as corpus positions
                       granularity=gran['fine'])
    if folds is None:
        folds = []
        for seed in args.seeds:
            split_seed = seed if args.vary_split else args.split_seed
            folds.append((f'seed{seed}', *random_split(len(data['fine']), split_seed)))
    # the published numbers are in-distribution; no reference for held-out papers
    ref = {h: f" (paper {PAPER_F1[gran[h]]})" if PAPER_F1[gran[h]] and not
           held_out_protocol(args) else '' for h in MT_HEADS}
    for run_id, tr, va, te in folds:
        if run_id in done:
            print(f"  {run_id}: already done, skipping")
            continue
        if args.max_train:
            tr = tr[:args.max_train]
        parts = {h: tuple([data[h][i] for i in part] for part in (tr, va, te)) for h in MT_HEADS}
        contexts = ({h: {name: part_contexts(data[h], part, load_full_texts(gran[h]))
                         for name, part in (('train', tr), ('val', va), ('test', te))}
                     for h in MT_HEADS} if getattr(args, 'context', 0) else None)
        seed = int(run_id[4:]) if run_id.startswith('seed') else args.seeds[0]
        seed += args.seed_offset
        start = time.time()
        records = train_multitask(seed, parts, labels, args, device, test_index=te,
                                  descriptions=descriptions, contexts=contexts)
        runtime = round(time.time() - start, 1)
        for head, res in records.items():
            res['run_id'] = run_id
            res['runtime_sec'] = runtime      # the shared training time
            if environment:
                res['environment'] = environment
            # a run found in one file only is redone: its old record is replaced
            results[head]['runs'] = [r for r in results[head]['runs']
                                     if r.get('run_id') != run_id] + [res]
        save()
        print(f"  {run_id}, {runtime:.0f}s: " + '; '.join(
            f"{h} test F1 {records[h]['f1']:.2f} (strict {records[h]['f1_strict']:.2f}), best "
            f"epoch {records[h]['best_epoch']}, running mean {results[h]['test_f1_mean']:.2f}"
            f"{ref[h]}" for h in MT_HEADS), flush=True)
    save()                         # every run_id of `folds` is now in both files
    for head in MT_HEADS:
        s = results[head]['summary']
        print(f"  {head}: micro F1 {s['micro']['mean']:.2f} 95% CI {s['micro']['ci95']}, "
              f"macro {s['macro']['mean']:.2f}, weighted {s['weighted']['mean']:.2f} "
              f"over {s['n_runs']} runs")
        print(f"Saved {paths[head]}")


def run_teachers(args, out_dir):
    """run() for --self-train --teachers N: one run per run_id, a results file per student seed.

    The files are those of runs with --tag m<offset> --seed-offset <offset> (offsets 0,
    100, ...; variant crf_st<N>_m<offset>), one per student seed: N, or --students M (the
    teachers keep the first N offsets). Each holds the students of its seed
    (train_with_teachers). All are checked (check_results_files) before training. A run
    trains the teachers and the students that its run_id lacks; only the files that got
    a new student are written, after every run.
    """
    granularities = granularities_of(args)
    tasks = {g: granularity_data(g, args) for g in granularities}
    sample, corpus = self_training_corpus(args.self_train, getattr(args, 'self_train_sample',
                                                                   SELF_TRAIN_SAMPLE))
    offsets = teacher_offsets(args.teachers)
    student_offsets = teacher_offsets(getattr(args, 'students', args.teachers))
    views = {off: SimpleNamespace(**{**vars(args), 'tag': f'm{off}', 'seed_offset': off})
             for off in student_offsets}
    for view in views.values():
        check_results_files(view, [(results_path(out_dir, g, view), tasks[g][1])
                                   for g in granularities], corpus['self_train_sha256'])
    device = torch_device(args)
    environment = run_environment(device)
    os.makedirs(out_dir, exist_ok=True)
    protocol = protocol_name(args)

    for granularity in granularities:
        data, labels, tie_info = tasks[granularity]
        paths = {off: results_path(out_dir, granularity, view) for off, view in views.items()}
        results = {}
        for off, view in views.items():
            variant = variant_name(view)
            results[off] = {'granularity': granularity, 'protocol': protocol, 'variant': variant,
                            'model': args.model, 'labels': labels, 'label_tie': tie_info,
                            'paper_test_f1': PAPER_F1[granularity],
                            'config': {**stored_config(view), **corpus},
                            'runs': resumable_runs(paths[off], labels, args.model, variant, off,
                                                   args.save_predictions,
                                                   corpus['self_train_sha256'])}
        done_by = {off: {r.get('run_id') for r in results[off]['runs']}
                   for off in student_offsets}
        done = set.intersection(*done_by.values())
        changed = set()            # the files that get a new student

        def save(offs):
            for off in offs:
                summarize(results[off])
                with open(paths[off], 'w') as f:
                    json.dump(results[off], f, indent=2)

        students = (f', {len(student_offsets)} students'
                    if len(student_offsets) != len(offsets) else '')
        print(f"\n=== NER {granularity}: {len(data)} sentences, {len(labels)} classes, "
              f"protocol {protocol}, {args.teachers} teachers (seed offsets {offsets})"
              f"{students}, device {device} ===")
        print(f"  self-training: {len(sample)} sentences sampled from {corpus['self_train']} "
              f"(sha256 {corpus['self_train_sha256'][:12]})")
        folds = make_folds(data, args, indices=True, granularity=granularity)
        if folds is None:
            folds = []
            for seed in args.seeds:
                split_seed = seed if args.vary_split else args.split_seed
                folds.append((f'seed{seed}', *random_split(len(data), split_seed)))
        for run_id, tr, va, te in folds:
            if run_id in done:
                print(f"  {run_id}: already done, skipping")
                continue
            train, val, test = ([data[i] for i in part] for part in (tr, va, te))
            if args.max_train:
                train = train[:args.max_train]
            seed = int(run_id[4:]) if run_id.startswith('seed') else args.seeds[0]
            # the students that this run_id lacks; the teachers are always trained again
            missing = [off for off in student_offsets if run_id not in done_by[off]]
            start = time.time()
            records = train_with_teachers(seed, offsets, train, val, test, labels, args, device,
                                          sample, test_index=te, student_offsets=missing)
            runtime = round(time.time() - start, 1)
            for off, res in records.items():
                res['run_id'] = run_id
                res['runtime_sec'] = runtime      # teachers and these students together
                if environment:
                    res['environment'] = environment
                results[off]['runs'].append(res)
                done_by[off].add(run_id)
            changed.update(records)
            save(records)
            print(f"  {run_id}, {runtime:.0f}s: students " + ', '.join(
                f"seed {records[off]['seed']} test F1 {records[off]['f1']:.2f}"
                for off in missing), flush=True)
        save(sorted(changed))
        for off in student_offsets:
            if results[off]['runs']:
                summary = summarize(results[off])['summary']
                print(f"  {results[off]['variant']}: micro F1 {summary['micro']['mean']:.2f} "
                      f"95% CI {summary['micro']['ci95']} over {summary['n_runs']} runs")
            print(f"{'Saved' if off in changed else 'Unchanged'} {paths[off]}")


def run_mt_teachers(args, out_dir):
    """run() for --multi-task --self-train --teachers N: one run per run_id, a fine and a
    coarse results file per student seed (variant crf_mt_st<N>_m<offset>).

    As run_teachers, with multi-task teachers and students (train_mt_with_teachers): a
    student counts as done only if both of its files hold the run, and a run trains the
    teachers and the students that its run_id lacks.
    """
    gran = head_granularities(args)
    data, labels, ties = {}, {}, {}
    for head in MT_HEADS:
        data[head], labels[head], ties[head] = granularity_data(gran[head], args)
    check_aligned(data['fine'], data['coarse'])
    sample, corpus = self_training_corpus(args.self_train, getattr(args, 'self_train_sample',
                                                                   SELF_TRAIN_SAMPLE))
    offsets = teacher_offsets(args.teachers)
    student_offsets = teacher_offsets(getattr(args, 'students', args.teachers))
    views = {off: SimpleNamespace(**{**vars(args), 'tag': f'm{off}', 'seed_offset': off})
             for off in student_offsets}
    paths = {(h, off): results_path(out_dir, gran[h], view) for off, view in views.items()
             for h in MT_HEADS}
    for off, view in views.items():
        check_results_files(view, [(paths[h, off], labels[h]) for h in MT_HEADS],
                            corpus['self_train_sha256'])
    device = torch_device(args)
    environment = run_environment(device)
    os.makedirs(out_dir, exist_ok=True)
    protocol = protocol_name(args)
    results = {}
    for off, view in views.items():
        variant = variant_name(view)
        for h in MT_HEADS:
            results[h, off] = {
                'granularity': gran[h], 'protocol': protocol, 'variant': variant,
                'model': args.model, 'labels': labels[h], 'label_tie': ties[h],
                'paper_test_f1': PAPER_F1[gran[h]],
                'config': {**stored_config(view), **corpus},
                'runs': resumable_runs(paths[h, off], labels[h], args.model, variant, off,
                                       args.save_predictions, corpus['self_train_sha256'])}
    done_by = {off: set.intersection(*({r.get('run_id') for r in results[h, off]['runs']}
                                       for h in MT_HEADS)) for off in student_offsets}
    done = set.intersection(*done_by.values())
    changed = set()

    def save(keys):
        for key in keys:
            summarize(results[key])
            with open(paths[key], 'w') as f:
                json.dump(results[key], f, indent=2)

    print(f"\n=== NER multi-task self-training: {len(data['fine'])} sentences, protocol "
          f"{protocol}, {args.teachers} teachers (seed offsets {offsets}), "
          f"{len(student_offsets)} students, device {device} ===")
    print(f"  self-training: {len(sample)} sentences sampled from {corpus['self_train']} "
          f"(sha256 {corpus['self_train_sha256'][:12]})")
    folds = make_folds(data['fine'], args, indices=True, granularity=gran['fine'])
    if folds is None:
        folds = []
        for seed in args.seeds:
            split_seed = seed if args.vary_split else args.split_seed
            folds.append((f'seed{seed}', *random_split(len(data['fine']), split_seed)))
    for run_id, tr, va, te in folds:
        if run_id in done:
            print(f"  {run_id}: already done, skipping")
            continue
        if args.max_train:
            tr = tr[:args.max_train]
        parts = {h: tuple([data[h][i] for i in part] for part in (tr, va, te)) for h in MT_HEADS}
        seed = int(run_id[4:]) if run_id.startswith('seed') else args.seeds[0]
        missing = [off for off in student_offsets if run_id not in done_by[off]]
        start = time.time()
        records = train_mt_with_teachers(seed, offsets, parts, labels, args, device, sample,
                                         test_index=te, student_offsets=missing)
        runtime = round(time.time() - start, 1)
        for off, by_head in records.items():
            for h, res in by_head.items():
                res['run_id'] = run_id
                res['runtime_sec'] = runtime      # teachers and these students together
                if environment:
                    res['environment'] = environment
                # a run found in one file only is redone: its old record is replaced
                results[h, off]['runs'] = [r for r in results[h, off]['runs']
                                           if r.get('run_id') != run_id] + [res]
            done_by[off].add(run_id)
            changed.update((h, off) for h in MT_HEADS)
        save([(h, off) for off in records for h in MT_HEADS])
        print(f"  {run_id}, {runtime:.0f}s: students " + ', '.join(
            f"seed {records[off]['fine']['seed']} fine {records[off]['fine']['f1']:.2f} "
            f"coarse {records[off]['coarse']['f1']:.2f}" for off in missing), flush=True)
    save(sorted(changed))
    for key in sorted(results):
        print(f"{'Saved' if key in changed else 'Unchanged'} {paths[key]}")


def tag_name(value):
    """argparse type of --tag: lowercase letters and digits, optionally then _m<N>.

    The tag goes into file names; <name>_m<N> names one seed of a family of runs
    (--crf --tag dapt2_m100: crf_dapt2_m100).
    """
    if not re.fullmatch(r'[a-z0-9]+(?:_m[0-9]+)?', value):
        raise argparse.ArgumentTypeError(f"invalid tag {value!r}: use lowercase letters and "
                                         f"digits, optionally followed by _m<N> (dapt2_m100)")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='MatSciBERT NER baseline')
    parser.add_argument('--granularity', choices=['fine', 'coarse', 'both', 'sofc',
                                                  'sofc_slot', 'sofc_both'],
                        default='both',
                        help="MaterioMiner's fine or coarse labels, or both; sofc: the SOFC-Exp "
                             "corpus (sofc_exp.py), its four entity types; sofc_slot: its slot "
                             "types; sofc_both: both SOFC-Exp levels")
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--vary-split', action='store_true',
                        help='new random split per seed')
    parser.add_argument('--leave-one-paper-out', action='store_true')
    parser.add_argument('--leave-two-papers-out', action='store_true',
                        default=argparse.SUPPRESS,
                        help='train on two papers and test on the other two, for each of the 6 '
                             'pairs (folds <paper>+<paper>): the inner folds of --entity-bias '
                             '(ROADMAP §4, week 7)')
    parser.add_argument('--published-split', action='store_true', default=argparse.SUPPRESS,
                        help="the train, validation and test sentences, and the tags, of the "
                             "MaterioMiner release (data/raw/dataset/published_split); every "
                             "--seeds value is a run on it with that model seed (ROADMAP §4, "
                             "week-5 amendment)")
    parser.add_argument('--papers', nargs='+', default=argparse.SUPPRESS, metavar='PAPER',
                        help='with --leave-one-paper-out: run only the folds that hold out these '
                             'papers (file names, .conll optional). The results file keeps '
                             'collecting folds, so the protocol can be run in parts; not stored '
                             'in the config')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--top-k', type=int, default=27,
                        help='fine-grained classes kept (MaterioMiner: 27)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch-size', type=int, default=16)
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument('--class-weights', action='store_true',
                         help='inverse sqrt-frequency loss weights (files tagged _cw)')
    variant.add_argument('--crf', action='store_true',
                         help='linear-chain CRF head with IOB2 constraints and Viterbi '
                              'decoding (files tagged _crf)')
    # absent unless given, so that the config stored by the other variants does not change
    parser.add_argument('--multi-task', action='store_true', default=argparse.SUPPRESS,
                        help='with --crf and --granularity both: one encoder with a fine and a '
                             'coarse CRF head, trained together; each run writes a fine and a '
                             'coarse results file (files tagged _crf_mt)')
    parser.add_argument('--label-semantics', action='store_true', default=argparse.SUPPRESS,
                        help='with --crf: score the tags of each label from an embedding of its '
                             'description (fine: MMO definition; coarse: the MMO classes '
                             'annotated with the type) instead of a linear classifier (files '
                             'tagged _crf_ls; with --multi-task, a head of this kind per '
                             'granularity, files tagged _crf_mt_ls)')
    parser.add_argument('--self-train', default=argparse.SUPPRESS, metavar='CORPUS',
                        help='with --crf: self-training. The CRF run is the teacher; it tags a '
                             'fixed sample of 2,000 sentences of CORPUS (one raw sentence per '
                             'line), and a student with the same seed trains on the gold data '
                             'plus the confident tagged sentences, their loss weighted 0.5 (files '
                             'tagged _crf_st)')
    parser.add_argument('--teachers', type=int, default=argparse.SUPPRESS, metavar='N',
                        help='with --self-train: N >= 2 CRF teachers with the model seeds of '
                             'seed offsets 0, 100, ..., 100 (N - 1); a silver sentence is one '
                             'that every teacher tags alike, each tag with a marginal of at '
                             'least 0.9 under every teacher, and every teacher seed trains a '
                             'student. One run writes N files, tagged m0, m100, ... (files '
                             'tagged _crf_st<N>_m<offset>); no --tag or --seed-offset')
    parser.add_argument('--context', type=int, default=argparse.SUPPRESS, metavar='N',
                        help='with --crf: document context (FLERT). Up to N subwords of the '
                             'neighbouring sentences of the same paper, and of the same part '
                             '(train, validation or test), on each side of a sentence in the '
                             'encoder input; only the sentence is tagged (files tagged _crf_ctx, '
                             '_crf_mt_ctx with --multi-task)')
    parser.add_argument('--augment-mentions', type=int, default=argparse.SUPPRESS, metavar='K',
                        help='with --crf: mention replacement (Dai and Adel, 2020). K copies '
                             'of each training sentence with an entity, each mention replaced '
                             f'with probability {MR_P} by a training mention of its type '
                             '(files tagged _crf_mr)')
    parser.add_argument('--entity-bias', type=float, nargs='+', default=argparse.SUPPRESS,
                        metavar='D',
                        help='with --crf: decode the test and validation sentences again with D '
                             'added to the emission of every tag but O, for each D (the run\'s '
                             'scores stay those of D = 0); per D the run stores the entity counts '
                             'of each test paper and of the validation set, and with '
                             '--save-predictions the test predictions (files tagged _bias; '
                             'ner_bias.py chooses a D)')
    parser.add_argument('--students', type=int, default=argparse.SUPPRESS, metavar='N',
                        help='with --teachers: N students, one per seed offset 0, 100, ..., '
                             '100 (N - 1), at least as many as the teachers, which keep the '
                             'first offsets and so the silver data. Students already in their '
                             'results files are not trained again (ROADMAP §4, week 5)')
    parser.add_argument('--self-train-sample', type=int, default=argparse.SUPPRESS, metavar='N',
                        help='with --self-train: the number of corpus sentences tagged (default '
                             '2,000; 0: the whole corpus)')
    parser.add_argument('--tag', type=tag_name, default=None, metavar='NAME',
                        help='appended to the variant name and the results file (crf + m100: '
                             'crf_m100; baseline + m100: m100); lowercase letters and digits, '
                             'optionally followed by _m<N> (crf + dapt2_m100: crf_dapt2_m100)')
    parser.add_argument('--seed-offset', type=int, default=0,
                        help='added to every model seed; the data split and the run ids do '
                             'not change, so runs still pair with the default ones (needs --tag)')
    parser.add_argument('--save-predictions', action='store_true',
                        help='store test_index, test_gold and test_pred (space-joined IOB2 '
                             'tags per test sentence) in every run')
    parser.add_argument('--max-train', type=int, default=0,
                        help='limit training sentences (smoke tests)')
    parser.add_argument('--overwrite', action='store_true',
                        help='replace the runs of a results file that this invocation cannot '
                             'resume (other labels, model, variant or seed offset; or no stored '
                             'predictions, with --save-predictions); without it, the script '
                             'stops before training. Smoke-test files are replaced regardless')
    parser.add_argument('--device', default=None)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--compare', nargs='?', const=os.path.join(PROCESSED_DIR, 'ner'),
                        default=None, metavar='DIR',
                        help='no training: pair every variant results file in DIR '
                             '(default: <PROCESSED_DIR>/ner) with its reference (the baseline; '
                             'crf / cw for tagged variants of those) and write '
                             'ner_comparisons.json')
    args = parser.parse_args(argv)
    if args.seed_offset and not args.tag:
        parser.error('--seed-offset needs --tag, so that offset runs get their own results file')
    if getattr(args, 'leave_two_papers_out', False):
        for option in ('vary_split', 'leave_one_paper_out', 'published_split'):
            if getattr(args, option, False):
                parser.error(f"--leave-two-papers-out is a protocol of its own: no "
                             f"--{option.replace('_', '-')}")
    if getattr(args, 'published_split', False):
        for option in ('vary_split', 'leave_one_paper_out', 'multi_task'):
            if option == 'multi_task' and args.granularity == 'sofc_both':
                continue                  # SOFC-Exp: one split by article for both levels
            if getattr(args, option, False):
                parser.error(f"--published-split is not available with "
                             f"--{option.replace('_', '-')} (its fine-grained and "
                             f"coarse-grained partitions differ)"
                             if option == 'multi_task' else
                             f"--published-split is a protocol of its own: no "
                             f"--{option.replace('_', '-')}")
    if getattr(args, 'papers', None) and not args.leave_one_paper_out:
        parser.error('--papers needs --leave-one-paper-out (it picks held-out papers)')
    if getattr(args, 'label_semantics', False):
        if not args.crf:
            parser.error('--label-semantics needs --crf (its scores are the CRF emissions)')
        if args.granularity in (*SOFC, 'sofc_both'):
            parser.error('--label-semantics has descriptions of the MaterioMiner types only')
    if getattr(args, 'self_train', None) is not None:
        if not args.crf:
            parser.error('--self-train needs --crf (silver sentences are chosen by CRF marginals)')
        if getattr(args, 'label_semantics', False):
            parser.error('--self-train is not available with --label-semantics')
        if getattr(args, 'multi_task', False) and not hasattr(args, 'teachers'):
            parser.error('--self-train with --multi-task needs --teachers (multi-task '
                         'self-training)')
        if not os.path.isfile(args.self_train):
            parser.error(f'--self-train: no such file: {args.self_train}')
    for option in ('teachers', 'self_train_sample'):
        if hasattr(args, option) and getattr(args, 'self_train', None) is None:
            parser.error(f"--{option.replace('_', '-')} needs --self-train")
    if getattr(args, 'teachers', 2) < 2:
        parser.error('--teachers: at least 2 (one teacher is --self-train alone)')
    if hasattr(args, 'teachers') and (args.tag or args.seed_offset):
        parser.error('--teachers writes its own seed tags (m0, m100, ...): no --tag or '
                     '--seed-offset')
    if hasattr(args, 'context'):
        if not args.crf:
            parser.error('--context needs --crf')
        if args.context < 1:
            parser.error('--context: at least 1 subword on each side')
        for option in ('label_semantics', 'self_train'):
            if getattr(args, option, None) not in (None, False):
                parser.error(f"--context is not available with --{option.replace('_', '-')}")
    if hasattr(args, 'augment_mentions'):
        if not args.crf:
            parser.error('--augment-mentions needs --crf')
        if args.augment_mentions < 1:
            parser.error('--augment-mentions: at least 1 copy')
        for option in ('multi_task', 'label_semantics', 'self_train', 'context'):
            if getattr(args, option, None) not in (None, False):
                parser.error(f"--augment-mentions is not available with "
                             f"--{option.replace('_', '-')}")
    if hasattr(args, 'entity_bias'):
        if not args.crf:
            parser.error('--entity-bias needs --crf (it shifts the emissions of the Viterbi '
                         'decode)')
        if getattr(args, 'self_train', None) is not None:
            parser.error('--entity-bias is not available with --self-train')
    if hasattr(args, 'students'):
        if not hasattr(args, 'teachers'):
            parser.error('--students needs --teachers')
        if args.students < args.teachers:
            parser.error('--students: at least as many as --teachers (every teacher seed '
                         'trains a student)')
    if getattr(args, 'self_train_sample', 0) < 0:
        parser.error('--self-train-sample: 0 (the whole corpus) or more sentences')
    if getattr(args, 'multi_task', False):
        if not args.crf:
            parser.error('--multi-task needs --crf (a CRF head per granularity)')
        if args.granularity not in GRANULARITY_PAIRS:
            parser.error('--multi-task trains the fine and the coarse head together: '
                         '--granularity must be both (the default) or sofc_both')
    # neither the tag nor the name in a seed tag (mt of mt_m100) may give a plain variant:
    # --crf --tag mt would write over crf_mt, --crf --tag mt_m100 over the multi-task crf_mt_m100
    stem = SimpleNamespace(**{**vars(args), 'tag': args.tag.split('_')[0]}) if args.tag else None
    if stem and variant_name(stem) in PLAIN_VARIANTS:
        parser.error(f'--tag {args.tag} would overwrite the results of the '
                     f'{variant_name(args)} variant')
    if min(args.seeds) + args.seed_offset < 0:
        parser.error('model seeds (seed + --seed-offset) must not be negative')
    return args


def main():
    args = parse_args()
    if args.compare:
        compare_all(args.compare)
        return
    try:
        run(args)
    except OverwriteError as e:
        sys.exit(f'error: {e}')


if __name__ == '__main__':
    main()
