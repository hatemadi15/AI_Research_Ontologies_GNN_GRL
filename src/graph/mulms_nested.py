"""
mulms_nested.py - MuLMS's nested NER, its published system re-run (SOTA plan P0)

Schrader et al. (2023) report MatSciBERT-CRF at 79.6 +- 0.4 micro F1 on MuLMS's test set:
one run per training configuration c = 1..5, which trains on the four train folds other than
train<c> and stops early on train<c>; every nested entity is scored. This module re-implements
that system from the release's code (github.com/boschresearch/mulms-wiesp2023, the files named
below; read, not copied):
  - data (source/ner/datasets/ner_bilou_dataset.py): the pinned parquet release (mulms.FILES,
    Hugging Face timo-pierre-schrader/MuLMS at mulms.REVISION). Per sentence: the release's
    tokens; its stacked BILOU tags, NER_labels_BILOU (up to three layers joined by '+', e.g.
    'B-VALUE+U-NUM', which the corpus reader made with the entities sorted by first token and
    then longest first, skipping an entity where three layers are full); and its gold
    entities, the set of (type, first token, last token) of NER_labels, all of them;
  - the model (source/ner/models/bert_crf.py): the encoder's last layer, dropout 0.2, a linear
    layer to the tags and a linear-chain CRF over them, with start and end scores and no
    constraints, initialised uniformly in [-0.1, 0.1], on each word's first sub-word. A word
    the tokenizer erases gets no tag, as in the release. The tags are the 473 stacked tags
    that occur in the release's data, 'O' first and the rest sorted (383 occur in its training
    folds; the release's constant mulms_all_bilou_tags lists 477, the 4 others never occur);
  - training (source/ner/approaches/ner_bilou_tagging.py, scripts/ner/run_ner_bilou_tagging.sh):
    AdamW with torch's defaults, 1e-4 for the encoder and the linear layer and 7e-3 for the
    CRF (the paper's Table 13 for MatSciBERT, the rates of its Table 11 row 'MuLMS 1e-4/7e-3:
    79.6 +- 0.4'; the script's default 5e-5 is SciBERT's), no schedule (the script leaves
    --enable_lr_decay off), no clipping, shuffled batches of 32, the CRF's negative
    log-likelihood summed over the batch (its gradient accumulated over micro-batches of 8
    sentences, the same update in less memory), at most 100 epochs. After each epoch the tune
    fold's micro F1 (`score`); the best model is kept, and training stops after 5 epochs
    without a gain (the code's patience). The paper's Appendix A also names an inverse
    square-root schedule and a patience of 15; --lr-schedule sqrt (the release's SqrtSchedule,
    its warm-up one epoch of steps, stepped after every batch) and --patience 15 run them;
  - scoring (source/ner/evaluation/metrics.py, calculate_prec_recall_f1_for_bilou): the
    predicted tags decoded layer by layer (`decode`) into (type, first, last) entities and
    compared per sentence as sets with the gold entities; micro F1 over the 13 types, and
    macro F1, their mean over the 13 (a type without predictions or gold counts 0). Dev is
    the release's 'validation' split.
One deviation: the release's predictions index the words the tokenizer keeps, so in a
sentence with an erased word the later predictions shift against the gold token indices;
here each prediction is put back at its word. Each run counts the sentences with an erased
word.

The CRF's forward pass is a product in log space (`StackedCRF.log_partition`), so its memory
grows with the number of tags, not its square: there are 473 stacked tags.

Outputs <out-dir>/ner_results_mulms_nested_configs_bilou_crf[_<tag>].json (<out-dir>:
<PROCESSED_DIR>/ner, as ner_baseline.py's): the settings, the
release's SHA-256 and one run per configuration (run_id c1..c5): the seed, the parts' sizes
and documents, the best epoch with the tune F1 of every epoch, the dev and test scores (micro
and macro F1, precision, recall, per type), the counts of erased and truncated words, the
environment, and with --save-predictions the test entities. A run already in the file is
skipped.

Usage:
  python src/graph/mulms_nested.py [--configs 1 2 3 4 5] [--seed 0] [--seed-offset N --tag NAME]
      [--model m3rg-iitd/matscibert] [--save-predictions] [--max-train N] [--offline]
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(os.environ.get('PROCESSED_DIR', ROOT / 'data' / 'processed')) / 'ner'
DEFAULT_MODEL = 'm3rg-iitd/matscibert'
PROTOCOL = 'nested_configs'
SCREEN_PROTOCOL = 'nested_screen'      # P1 (ROADMAP §4): scored on the dev set only
SYSTEM = 'bilou_crf'
SPAN_SYSTEM = 'span'                   # P1's E: span_ner's span head
CONFIGS = (1, 2, 3, 4, 5)
TYPES = ('MAT', 'NUM', 'VALUE', 'UNIT', 'PROPERTY', 'FORM', 'MEASUREMENT', 'CITE', 'SAMPLE',
         'TECHNIQUE', 'DEV', 'RANGE', 'INSTRUMENT')     # the release's mulms_named_entities
# the release's settings (scripts/ner/run_ner_bilou_tagging.sh and the training loop)
SETTINGS = {'lr': 1e-4, 'lr_crf': 7e-3, 'batch_size': 32, 'epochs': 100, 'patience': 5,
            'dropout': 0.2, 'crf_init': 0.1, 'max_length': 512, 'micro_batch': 8}
SPLITS = {'train': 'train-00000-of-00001.parquet',
          'validation': 'validation-00000-of-00001.parquet',
          'test': 'test-00000-of-00001.parquet'}


# ---------------------------------------------------------------------------------------------
# The release's data and its configurations
# ---------------------------------------------------------------------------------------------
def sentences_of(rows):
    """The release's rows as sentences: {doc_id, split, tokens, tags, gold}, gold the set of
    (type, first, last) of NER_labels. A row whose tags and tokens differ in length stops the
    conversion."""
    out = []
    for r in rows:
        tokens, tags = list(r['tokens']), list(r['NER_labels_BILOU'])
        if len(tokens) != len(tags):
            raise ValueError(f"{r['doc_id']}: {len(tokens)} tokens, {len(tags)} tags")
        labels = r['NER_labels'] or {}
        gold = {(t, int(i[0]), int(i[1]))
                for t, i in zip(labels.get('value', []), labels.get('tokenIndices', []))}
        out.append({'doc_id': r['doc_id'], 'split': r['data_split'], 'tokens': tokens,
                    'tags': tags, 'gold': gold})
    return out


def read_release(cache=None, offline=False):
    """The sentences of the pinned parquet release (mulms.fetch_sources checks the SHA-256 of
    every file), in the files' order: train, validation, test."""
    import pyarrow.parquet as pq
    import mulms
    paths = mulms.fetch_sources(Path(cache) if cache else mulms.CACHE_DIR, offline=offline)
    rows = []
    for name in SPLITS.values():
        rows += pq.read_table(paths[f'MuLMS_Corpus/{name}']).to_pylist()
    return sentences_of(rows)


def configuration(sentences, c):
    """{train, tune, dev, test} of configuration c: training on the train folds other than
    train<c>, tuning (early stopping) on train<c>, the release's dev and test sets."""
    if c not in CONFIGS:
        raise ValueError(f'configuration {c}: not one of {CONFIGS}')
    parts = {'train': [], 'tune': [], 'dev': [], 'test': []}
    for s in sentences:
        split = s['split']
        if split == f'train{c}':
            parts['tune'].append(s)
        elif split.startswith('train'):
            parts['train'].append(s)
        elif split == 'dev':
            parts['dev'].append(s)
        elif split == 'test':
            parts['test'].append(s)
        else:
            raise ValueError(f"{s['doc_id']}: split {split}")
    return parts


def configurations_record(sentences):
    """{c<k>: {part: sorted documents}} of the 5 configurations, for the data file."""
    return {f'c{c}': {part: sorted({s['doc_id'] for s in group})
                      for part, group in configuration(sentences, c).items()}
            for c in CONFIGS}


def outermost_iob2(sentence):
    """IOB2 tags of a sentence's outermost entities (P1's C2 joins them to SOFC-Exp): the gold
    entities by first word, the longer first, then by type; an entity is kept unless it
    overlaps one kept before it."""
    tags = ['O'] * len(sentence['tokens'])
    taken = set()
    for kind, i, j in sorted(sentence['gold'], key=lambda e: (e[1], e[1] - e[2], e[0])):
        if taken.isdisjoint(range(i, j + 1)):
            tags[i] = f'B-{kind}'
            for k in range(i + 1, j + 1):
                tags[k] = f'I-{kind}'
            taken.update(range(i, j + 1))
    return tags


def part_contexts(group):
    """P1's C3: per sentence of a part, (left, right) context sentences as word lists, nearest
    first: the other sentences of its document within the part, in the release's order (as
    ner_baseline.part_contexts; a MuLMS document lies in one part)."""
    by_doc = {}
    for k, s in enumerate(group):
        by_doc.setdefault(s['doc_id'], []).append(k)
    out = []
    for k, s in enumerate(group):
        idx = by_doc[s['doc_id']]
        at = idx.index(k)
        out.append(([group[j]['tokens'] for j in reversed(idx[:at])],
                    [group[j]['tokens'] for j in idx[at + 1:]]))
    return out


def sofc_joint_sentences():
    """P1's C2 for MuLMS: SOFC-Exp's 34 training and validation articles' sentences as
    (tokens, IOB2 tags of its 4 entity types), without the articles that MuLMS also holds
    (results/sota/shared_documents.json), and its types."""
    import ner_baseline as nb
    import sota_p1
    shared = sota_p1.shared_sofc_articles()
    split = nb.sofc_split()
    data = nb.load_split_corpus('sofc')
    labels, _ = nb.select_labels(data)
    out = [(list(tokens), list(tags)) for tokens, tags, doc in data
           if split[doc.removesuffix('.conll')] in ('train', 'dev')
           and doc.removesuffix('.conll') not in shared]
    return out, list(labels)


def tag_vocabulary(sentences):
    """Every stacked tag of the release, 'O' first and the rest sorted."""
    tags = {t for s in sentences for t in s['tags']}
    return ['O'] + sorted(tags - {'O'})


# ---------------------------------------------------------------------------------------------
# The release's decoder and scorer
# ---------------------------------------------------------------------------------------------
def decode(tags):
    """The entities of one sentence's stacked tags, decoded as the release's
    calculate_prec_recall_f1_for_bilou does: each word's tags split at '+' into a stack; the
    words are visited in order, each time the first tag of the first non-empty stack is
    taken. 'O' empties its word's stack; U-T gives (T, i, i); B-T takes, from the next words
    in order, I-T tags at the front of their stacks until an L-T, which ends the entity (T, i,
    j) (a stack that is empty or opens with another tag ends the search without an entity),
    and the visit starts again from the first word; any other tag is dropped. Returns a set of
    (type, first, last)."""
    stacks = [t.split('+') for t in tags]
    found = set()
    while any(stacks):
        for i in range(len(stacks)):
            if not stacks[i]:
                continue
            first = stacks[i][0]
            kind = first.split('-')[-1]
            if kind == 'O':
                stacks[i] = []
            elif first == f'U-{kind}':
                found.add((kind, i, i))
                stacks[i] = stacks[i][1:]
            elif first == f'B-{kind}':
                stacks[i] = stacks[i][1:]
                valid, j = False, i
                for j in range(i + 1, len(stacks)):
                    if not stacks[j]:
                        break
                    if stacks[j][0] == f'I-{kind}':
                        stacks[j] = stacks[j][1:]
                    elif stacks[j][0] == f'L-{kind}':
                        valid = True
                        stacks[j] = stacks[j][1:]
                        break
                    else:
                        break
                if valid:
                    found.add((kind, i, j))
                break                               # the visit starts again from the first word
            else:
                stacks[i] = stacks[i][1:]
    return found


def score(predicted, gold, types=TYPES):
    """The release's scores of per-sentence entity sets: TP, FP and FN per type from set
    intersections; micro P, R and F1 over all types; macro F1, the mean F1 over `types` (0
    for a type with nothing predicted or nothing gold). Percentages, rounded to 2 places."""
    if len(predicted) != len(gold):
        raise ValueError(f'{len(predicted)} predictions for {len(gold)} sentences')
    counts = {t: {'TP': 0, 'FP': 0, 'FN': 0} for t in types}
    for pred, true in zip(predicted, gold):
        for kind, entities in (('TP', pred & true), ('FP', pred - true), ('FN', true - pred)):
            for e in entities:
                counts[e[0]][kind] += 1

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)
    per_type = {}
    for t, c in counts.items():
        p, r, f = prf(c['TP'], c['FP'], c['FN'])
        per_type[t] = {'precision': round(100 * p, 2), 'recall': round(100 * r, 2),
                       'f1': round(100 * f, 2), **c}
    total = {k: sum(c[k] for c in counts.values()) for k in ('TP', 'FP', 'FN')}
    p, r, f = prf(total['TP'], total['FP'], total['FN'])
    macro = sum(prf(c['TP'], c['FP'], c['FN'])[2] for c in counts.values()) / len(types)
    return {'micro_f1': round(100 * f, 2), 'precision': round(100 * p, 2),
            'recall': round(100 * r, 2), 'macro_f1': round(100 * macro, 2), **total,
            'per_type': per_type}


# ---------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------
def _define_modules():
    import torch

    class StackedCRF(torch.nn.Module):
        """A linear-chain CRF with start and end scores and no constraints. Its forward
        recursion is a product in log space: log sum_i exp(a_i + T_ij) = m + c_j +
        log(exp(a - m) @ exp(T - c))_j, with m the row maximum of a and c_j the column
        maximum of T, so its memory is linear in the number of tags."""

        def __init__(self, n, init=SETTINGS['crf_init']):
            super().__init__()
            self.transitions = torch.nn.Parameter(torch.empty(n, n).uniform_(-init, init))
            self.start = torch.nn.Parameter(torch.empty(n).uniform_(-init, init))
            self.end = torch.nn.Parameter(torch.empty(n).uniform_(-init, init))

        def log_partition(self, emissions, mask):
            """log Z per sequence; emissions (B, W, L), mask (B, W) left-aligned."""
            col = self.transitions.max(dim=0, keepdim=True).values          # (1, L)
            expt = torch.exp(self.transitions - col)
            alpha = self.start + emissions[:, 0]
            for t in range(1, emissions.size(1)):
                m = alpha.max(dim=1, keepdim=True).values
                prod = torch.exp(alpha - m) @ expt
                nxt = torch.log(prod.clamp_min(torch.finfo(prod.dtype).tiny)) + m + col \
                    + emissions[:, t]
                alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
            return torch.logsumexp(alpha + self.end, dim=1)

        def gold_score(self, emissions, tags, mask):
            score = self.start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, :1]).squeeze(1)
            for t in range(1, emissions.size(1)):
                step = (self.transitions[tags[:, t - 1], tags[:, t]]
                        + emissions[:, t].gather(1, tags[:, t:t + 1]).squeeze(1))
                score = score + step * mask[:, t].to(step.dtype)
            last = tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
            return score + self.end[last]

        def nll(self, emissions, tags, mask):
            """The negative log-likelihood summed over the batch (the release's reduction)."""
            return (self.log_partition(emissions, mask)
                    - self.gold_score(emissions, tags, mask)).sum()

        @torch.no_grad()
        def decode(self, emissions, mask):
            """Viterbi tag-id paths, one list per sequence."""
            score = self.start + emissions[:, 0]
            back = []
            for t in range(1, emissions.size(1)):
                best, idx = (score.unsqueeze(2) + self.transitions).max(dim=1)
                score = torch.where(mask[:, t].unsqueeze(1), best + emissions[:, t], score)
                back.append(idx)
            score = score + self.end
            paths = []
            for b, length in enumerate(mask.sum(1).tolist()):
                path = [int(score[b].argmax())]
                for t in range(length - 1, 0, -1):
                    path.append(int(back[t - 1][b, path[-1]]))
                paths.append(path[::-1])
            return paths

    class NestedTagger(torch.nn.Module):
        """The encoder, dropout, a linear layer and the CRF, on each kept word's first
        sub-word."""

        def __init__(self, encoder, n_tags, dropout=SETTINGS['dropout']):
            super().__init__()
            self.encoder = encoder
            self.dropout = torch.nn.Dropout(dropout)
            self.linear = torch.nn.Linear(encoder.config.hidden_size, n_tags)
            self.crf = StackedCRF(n_tags)

        def emissions(self, batch):
            hidden = self.encoder(input_ids=batch['input_ids'],
                                  attention_mask=batch['attention_mask']).last_hidden_state
            logits = self.linear(self.dropout(hidden))
            index = batch['first'].clamp(min=0)
            return logits.gather(1, index.unsqueeze(2).expand(-1, -1, logits.size(2)))

    return StackedCRF, NestedTagger


# ---------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------
def features(sentence, tokenizer, tag_index, max_length=SETTINGS['max_length'], left=(),
             right=(), tags=None):
    """{input_ids, first, kept, tags, erased, truncated}: the sub-words of a sentence, the
    position of each kept word's first sub-word, the kept words' positions and tag ids, and
    the numbers of words with no sub-word and of words cut by the length limit. left and
    right (P1's C3): context words around the sentence, never tagged. tags: the sentence's
    tags when not its own (P1's C2: SOFC-Exp's)."""
    words = list(left) + list(sentence['tokens']) + list(right)
    lo, hi = len(left), len(left) + len(sentence['tokens'])
    enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length)
    first = {}
    for pos, w in enumerate(enc.word_ids()):
        if w is not None and lo <= w < hi and w - lo not in first:
            first[w - lo] = pos
    full = tokenizer(sentence['tokens'], is_split_into_words=True, truncation=False)
    present = {w for w in full.word_ids() if w is not None}
    kept = sorted(first)
    tags = sentence['tags'] if tags is None else tags
    return {'input_ids': enc['input_ids'], 'first': [first[w] for w in kept], 'kept': kept,
            'tags': [tag_index[tags[w]] for w in kept],
            'erased': len(sentence['tokens']) - len(present),
            'truncated': len(present) - len(kept)}


def collate(items, pad_id, device):
    import torch
    width = max(len(f['input_ids']) for f in items)
    words = max(max(len(f['first']) for f in items), 1)
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    att = torch.zeros((len(items), width), dtype=torch.long)
    first = torch.full((len(items), words), -1, dtype=torch.long)
    tags = torch.zeros((len(items), words), dtype=torch.long)
    for b, f in enumerate(items):
        ids[b, :len(f['input_ids'])] = torch.tensor(f['input_ids'])
        att[b, :len(f['input_ids'])] = 1
        first[b, :len(f['first'])] = torch.tensor(f['first'], dtype=torch.long)
        tags[b, :len(f['tags'])] = torch.tensor(f['tags'], dtype=torch.long)
    mask = first >= 0
    return {'input_ids': ids.to(device), 'attention_mask': att.to(device),
            'first': first.to(device), 'tags': tags.to(device), 'mask': mask.to(device)}


def predict(model, feats, sentences, vocab, device, batch_size):
    """The decoded entities of each sentence: the kept words' Viterbi tags put back at their
    words ('O' at an erased or cut word), then `decode`."""
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(feats), batch_size):
            items = feats[start:start + batch_size]
            batch = collate(items, model.pad_id, device)
            if not batch['mask'].any():
                out += [set() for _ in items]
                continue
            emissions = model.emissions(batch)
            mask = batch['mask'].clone()
            mask[:, 0] = True                         # a sentence without a kept word decodes to O
            paths = model.crf.decode(emissions, mask)
            for f, s, path in zip(items, sentences[start:start + batch_size], paths):
                tags = ['O'] * len(s['tokens'])
                for w, t in zip(f['kept'], path[:len(f['kept'])]):
                    tags[w] = vocab[t]
                out.append(decode(tags))
    return out


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------
def train_config(sentences, c, seed, args, device, vocab):
    """One run: configuration c at `seed`; returns its record. P1's options (ROADMAP §4) on
    top of P0's run, which they leave as it was when none is set: --screen (no test sentence
    is encoded or scored; the record has the dev scores only), --context N (C3: up to N
    sub-words of the document's other sentences of the same part on each side, untagged),
    --joint-sofc (C2: SOFC-Exp's sentences on a CRF head of their own, each epoch min(n, m) of
    them drawn by random.Random(f'{seed}-{epoch}') and shuffled in with MuLMS's; never scored)
    and --gradient-checkpointing; --span-head runs train_span_config instead."""
    import torch
    from transformers import AutoModel, set_seed
    import ner_baseline as nb
    if getattr(args, 'span_head', False):
        return train_span_config(sentences, c, seed, args, device)
    screen = getattr(args, 'screen', False)
    parts = configuration(sentences, c)
    if screen:
        del parts['test']
    if args.max_train:
        parts = {**parts, 'train': parts['train'][:args.max_train],
                 'tune': parts['tune'][:args.max_train]}
    set_seed(seed)
    tokenizer = nb.load_tokenizer(args.model)
    index = {t: i for i, t in enumerate(vocab)}
    feats = {name: encode_part(group, tokenizer, index, args) for name, group in parts.items()}
    _, NestedTagger = _define_modules()
    encoder = AutoModel.from_pretrained(args.model, dtype=torch.float32)
    model = NestedTagger(encoder, len(vocab)).to(device)
    model.pad_id = tokenizer.pad_token_id
    joint = None
    if getattr(args, 'joint_sofc', False):
        StackedCRF, _ = _define_modules()
        other, other_types = sofc_joint_sentences()
        other_vocab = ['O'] + [f'{p}-{k}' for k in other_types for p in ('B', 'I')]
        other_index = {k: i for i, k in enumerate(other_vocab)}
        model.sofc_linear = torch.nn.Linear(encoder.config.hidden_size, len(other_vocab))
        model.sofc_crf = StackedCRF(len(other_vocab))
        model.to(device)
        joint = {'feats': [features({'tokens': tokens, 'tags': tags}, tokenizer, other_index)
                           for tokens, tags in other], 'types': other_types}
    if getattr(args, 'gradient_checkpointing', False):
        encoder.gradient_checkpointing_enable()
    crf_names = ('crf.', 'sofc_crf.')
    lm = [p for n, p in model.named_parameters() if not n.startswith(crf_names)]
    lm_opt = torch.optim.AdamW(lm, lr=args.lr)
    crf_opt = torch.optim.AdamW([p for n, p in model.named_parameters()
                                 if n.startswith(crf_names)], lr=args.lr_crf)
    rng = random.Random(seed)
    order = [i for i, f in enumerate(feats['train']) if f['first']]
    schedules = []
    if args.lr_schedule == 'sqrt':
        warmup = math.ceil(len(order) / args.batch_size)
        schedules = [torch.optim.lr_scheduler.LambdaLR(opt, sqrt_schedule(warmup))
                     for opt in (lm_opt, crf_opt)]
    gold = {name: [s['gold'] for s in group] for name, group in parts.items()}

    def sofc_nll(items):
        batch = collate(items, model.pad_id, device)
        hidden = model.encoder(input_ids=batch['input_ids'],
                               attention_mask=batch['attention_mask']).last_hidden_state
        logits = model.sofc_linear(model.dropout(hidden))
        emissions = logits.gather(1, batch['first'].clamp(min=0).unsqueeze(2).expand(
            -1, -1, logits.size(2)))
        return model.sofc_crf.nll(emissions, batch['tags'], batch['mask'])

    best, best_f1, best_epoch, waited, history = None, -1.0, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if joint is None:
            rng.shuffle(order)                        # P0's order of operations
        units = [('mulms', i) for i in order]
        if joint is not None:
            n_other = min(len(order), len(joint['feats']))
            pick = random.Random(f'{seed}-{epoch}').sample(range(len(joint['feats'])), n_other)
            units += [('sofc', j) for j in pick]
            rng.shuffle(units)
        for start in range(0, len(units), args.batch_size):
            batch_units = units[start:start + args.batch_size]
            lm_opt.zero_grad()
            crf_opt.zero_grad()
            # the batch's summed loss, its gradient accumulated over micro-batches (memory)
            mulms_ids = [i for corpus, i in batch_units if corpus == 'mulms']
            for m in range(0, len(mulms_ids), args.micro_batch):
                micro = collate([feats['train'][i] for i in mulms_ids[m:m + args.micro_batch]],
                                model.pad_id, device)
                model.crf.nll(model.emissions(micro), micro['tags'], micro['mask']).backward()
            sofc_ids = [j for corpus, j in batch_units if corpus == 'sofc']
            for m in range(0, len(sofc_ids), args.micro_batch):
                sofc_nll([joint['feats'][j] for j in sofc_ids[m:m + args.micro_batch]]).backward()
            lm_opt.step()
            crf_opt.step()
            for schedule in schedules:
                schedule.step()
        tune = score(predict(model, feats['tune'], parts['tune'], vocab, device,
                             args.batch_size), gold['tune'])
        history.append({'epoch': epoch, 'tune_micro_f1': tune['micro_f1'],
                        'tune_macro_f1': tune['macro_f1']})
        print(f"  c{c} epoch {epoch}: tune micro F1 {tune['micro_f1']:.2f}", flush=True)
        if tune['micro_f1'] > best_f1:
            best_f1, best_epoch, waited = tune['micro_f1'], epoch, 0
            best = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited == args.patience:
                break
    model.load_state_dict(best)
    record = run_record(c, seed, parts, best_epoch, best_f1, history, feats, screen)
    if joint is not None:
        record['joint'] = {'n_other': len(joint['feats']), 'other_types': joint['types'],
                           'other_per_epoch': min(len(order), len(joint['feats']))}
    for name in ('dev',) if screen else ('dev', 'test'):
        predicted = predict(model, feats[name], parts[name], vocab, device, args.batch_size)
        record[name] = score(predicted, gold[name])
        if name == 'test' and args.save_predictions:
            record['test_pred'] = [sorted(list(e) for e in p) for p in predicted]
    if not screen:
        record['f1'] = record['test']['micro_f1']
    environment = nb.run_environment(str(device))
    if environment:
        record['environment'] = environment
    return record


def encode_part(group, tokenizer, index, args, tags=None):
    """features() of every sentence of a part, with --context's words around each (C3)."""
    import ner_baseline as nb
    n = getattr(args, 'context', 0)
    if not n:
        return [features(s, tokenizer, index) for s in group]
    out = []
    for s, (left, right) in zip(group, part_contexts(group)):
        own = len(tokenizer(s['tokens'], is_split_into_words=True,
                            truncation=False)['input_ids'])
        words = nb.context_words(tokenizer, left, right,
                                 min(n, max(0, (SETTINGS['max_length'] - own) // 2)))
        out.append(features(s, tokenizer, index, left=words[0], right=words[1]))
    return out


def run_record(c, seed, parts, best_epoch, best_f1, history, feats, screen):
    """The fields of a run's record before its scores (screen: no test field)."""
    record = {'run_id': f'c{c}', 'config': c, 'seed': seed, 'tune_split': f'train{c}',
              'n_train': len(parts['train']), 'n_tune': len(parts['tune']),
              'n_dev': len(parts['dev'])}
    if not screen:
        record['n_test'] = len(parts['test'])
    record.update({'best_epoch': best_epoch, 'tune_micro_f1': best_f1, 'history': history,
                   'erased_word_sentences': {name: sum(1 for f in group if f['erased'])
                                             for name, group in feats.items()},
                   'truncated_words': sum(f['truncated'] for group in feats.values()
                                          for f in group)})
    return record


def train_span_config(sentences, c, seed, args, device):
    """P1's E on MuLMS (--span-head): span_ner's span head on the encoder, by P0's recipe
    (AdamW with torch's defaults, the encoder at args.lr and the head at span_ner.HEAD_LR, no
    schedule unless --lr-schedule sqrt, no clipping, shuffled batches of args.batch_size in
    micro-batches, the tune fold's micro F1 picking the epoch, args.patience), decoded nested.
    The loss is the binary cross-entropy averaged over the batch's (span, type) pairs. The
    width cap comes from the training folds' entities."""
    import torch
    from transformers import AutoModel, set_seed
    import ner_baseline as nb
    import span_ner
    screen = getattr(args, 'screen', False)
    parts = configuration(sentences, c)
    if screen:
        del parts['test']
    if args.max_train:
        parts = {**parts, 'train': parts['train'][:args.max_train],
                 'tune': parts['tune'][:args.max_train]}
    set_seed(seed)
    tokenizer = nb.load_tokenizer(args.model)
    types = list(TYPES)
    type_index = {k: i for i, k in enumerate(types)}
    spans = {name: [{'tokens': s['tokens'], 'entities': s['gold']} for s in group]
             for name, group in parts.items()}
    cap = span_ner.width_cap(spans['train'])
    contexts = {name: part_contexts(group) for name, group in parts.items()} \
        if getattr(args, 'context', 0) else None

    def encoded(name):
        out = []
        for k, s in enumerate(spans[name]):
            left = right = ()
            if contexts is not None:
                own = len(tokenizer(s['tokens'], is_split_into_words=True,
                                    truncation=False)['input_ids'])
                left, right = nb.context_words(tokenizer, *contexts[name][k], min(
                    args.context, max(0, (SETTINGS['max_length'] - own) // 2)))
            out.append(span_ner.encode(s, tokenizer, type_index, cap, SETTINGS['max_length'],
                                       left, right))
        return out
    feats = {name: encoded(name) for name in parts}
    encoder = AutoModel.from_pretrained(args.model, dtype=torch.float32)
    model = span_ner.make_tagger(encoder, len(types), cap).to(device)
    if getattr(args, 'gradient_checkpointing', False):
        encoder.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW([{'params': list(encoder.parameters())},
                                   {'params': list(model.head.parameters()),
                                    'lr': span_ner.HEAD_LR}], lr=args.lr)
    schedules = []
    order = [i for i, f in enumerate(feats['train']) if f['spans']]
    if args.lr_schedule == 'sqrt':
        schedules = [torch.optim.lr_scheduler.LambdaLR(
            optimizer, sqrt_schedule(math.ceil(len(order) / args.batch_size)))]
    rng = random.Random(seed)
    pad_id = tokenizer.pad_token_id

    def predicted(name):
        return span_ner.predict(model, feats[name], types, True, args.batch_size, pad_id,
                                device)
    best, best_f1, best_epoch, waited, history = None, -1.0, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            pairs = sum(len(feats['train'][i]['spans']) for i in ids) * len(types)
            optimizer.zero_grad()
            for m in range(0, len(ids), args.micro_batch):
                items = [feats['train'][i] for i in ids[m:m + args.micro_batch]]
                batch = span_ner.collate(items, pad_id, len(types), device)
                logits = model(batch)
                mask = batch['span_mask'].unsqueeze(2).expand_as(logits)
                (torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[mask], batch['targets'][mask], reduction='sum') / pairs).backward()
            optimizer.step()
            for schedule in schedules:
                schedule.step()
        tune = score(predicted('tune'), [s['gold'] for s in parts['tune']])
        history.append({'epoch': epoch, 'tune_micro_f1': tune['micro_f1'],
                        'tune_macro_f1': tune['macro_f1']})
        print(f"  c{c} epoch {epoch}: tune micro F1 {tune['micro_f1']:.2f}", flush=True)
        if tune['micro_f1'] > best_f1:
            best_f1, best_epoch, waited = tune['micro_f1'], epoch, 0
            best = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited == args.patience:
                break
    model.load_state_dict(best)
    record = run_record(c, seed, parts, best_epoch, best_f1, history, feats, screen)
    record['span_head'] = {'width_cap': cap, 'head_lr': span_ner.HEAD_LR,
                           'unreachable': {name: sum(f['unreachable'] for f in group)
                                           for name, group in feats.items()}}
    for name in ('dev',) if screen else ('dev', 'test'):
        pred = predicted(name)
        record[name] = score(pred, [s['gold'] for s in parts[name]])
        if name == 'test' and args.save_predictions:
            record['test_pred'] = [sorted(list(e) for e in p) for p in pred]
    if not screen:
        record['f1'] = record['test']['micro_f1']
    environment = nb.run_environment(str(device))
    if environment:
        record['environment'] = environment
    return record


def sqrt_schedule(warmup):
    """The release's SqrtSchedule as a LambdaLR factor: 0 at step 0, then sqrt(w) *
    min(step ** -0.5, step * w ** -1.5), a linear warm-up over w steps and an inverse
    square-root decay."""
    def factor(step):
        if step == 0:
            return 0.0
        return math.sqrt(warmup) * min(step ** -0.5, step * warmup ** -1.5)
    return factor


def results_path(out_dir, tag=None, max_train=0, screen=False, span=False):
    name = (f"ner_results_mulms_{SCREEN_PROTOCOL if screen else PROTOCOL}_"
            f"{SPAN_SYSTEM if span else SYSTEM}")
    name += f'_{tag}' if tag else ''
    name += f'_smoke{max_train}' if max_train else ''
    return Path(out_dir) / f'{name}.json'


def summarize(results):
    import statistics
    runs = results['runs']
    for name in ('dev',) if results.get('screen') else ('test', 'dev'):
        values = [r[name]['micro_f1'] for r in runs]
        results[f'{name}_micro_f1_mean'] = round(statistics.fmean(values), 2) if values else None
        results[f'{name}_micro_f1_sd'] = (round(statistics.stdev(values), 2)
                                          if len(values) > 1 else None)


def run(args):
    import torch
    import mulms
    import ner_baseline as nb
    sentences = read_release(args.cache, args.offline)
    vocab = tag_vocabulary(sentences)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device.startswith('cuda'):
        nb.cuda_determinism()
    if args.threads:
        torch.set_num_threads(args.threads)
    path = results_path(args.out_dir, args.tag, args.max_train, args.screen, args.span_head)
    path.parent.mkdir(parents=True, exist_ok=True)
    p1 = {k: getattr(args, k) for k in P1_OPTIONS if getattr(args, k)}   # P1's options set
    results = json.loads(path.read_text()) if path.exists() and not args.overwrite else {
        'corpus': 'mulms', 'protocol': SCREEN_PROTOCOL if args.screen else PROTOCOL,
        'system': SPAN_SYSTEM if args.span_head else SYSTEM, 'model': args.model,
        'release': {'revision': mulms.REVISION, 'files': dict(mulms.FILES)},
        'config': {**{k: getattr(args, k) for k in ('lr', 'lr_crf', 'batch_size', 'epochs',
                                                     'patience', 'lr_schedule', 'micro_batch',
                                                     'seed', 'seed_offset', 'tag',
                                                     'max_train')},
                   'dropout': SETTINGS['dropout'], 'crf_init': SETTINGS['crf_init'],
                   'max_length': SETTINGS['max_length'], 'n_tags': len(vocab), **p1},
        **({'screen': True} if args.screen else {}),
        'configurations': configurations_record(sentences), 'runs': []}
    done = {r['run_id'] for r in results['runs']}
    for c in args.configs:
        if f'c{c}' in done:
            print(f'  c{c}: already done, skipping')
            continue
        start = time.time()
        record = train_config(sentences, c, args.seed + args.seed_offset, args, device, vocab)
        record['runtime_sec'] = round(time.time() - start, 1)
        results['runs'].append(record)
        summarize(results)
        path.write_text(json.dumps(results, indent=2))
        if args.screen:
            print(f"  c{c}: dev micro F1 {record['dev']['micro_f1']:.2f} (best epoch "
                  f"{record['best_epoch']})")
        else:
            print(f"  c{c}: test micro F1 {record['test']['micro_f1']:.2f}, dev "
                  f"{record['dev']['micro_f1']:.2f} (best epoch {record['best_epoch']})")
    print(f'wrote {path}')
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MuLMS's nested NER, the published system")
    parser.add_argument('--configs', type=int, nargs='+', default=list(CONFIGS),
                        choices=CONFIGS)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--lr', type=float, default=SETTINGS['lr'])
    parser.add_argument('--lr-crf', type=float, default=SETTINGS['lr_crf'])
    parser.add_argument('--batch-size', type=int, default=SETTINGS['batch_size'])
    parser.add_argument('--micro-batch', type=int, default=SETTINGS['micro_batch'],
                        help="sentences per forward pass; a batch's gradient is summed over "
                             'its micro-batches, so the update is the batch\'s')
    parser.add_argument('--epochs', type=int, default=SETTINGS['epochs'])
    parser.add_argument('--patience', type=int, default=SETTINGS['patience'])
    parser.add_argument('--lr-schedule', choices=('none', 'sqrt'), default='none',
                        help="none (the released script) or sqrt (the paper's Appendix A)")
    parser.add_argument('--save-predictions', action='store_true')
    # the SOTA plan's P1 (ROADMAP §4, the screens)
    parser.add_argument('--screen', action='store_true',
                        help='P1: no test sentence is encoded or scored; the record has the '
                             'dev scores (protocol nested_screen)')
    parser.add_argument('--context', type=int, default=0, metavar='N',
                        help="P1's C3: up to N sub-words of document context on each side")
    parser.add_argument('--joint-sofc', action='store_true',
                        help="P1's C2: joint training with SOFC-Exp's training and validation "
                             'articles on a CRF head of their own')
    parser.add_argument('--span-head', action='store_true',
                        help="P1's E: span_ner's span head instead of the CRF (system span)")
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help="P1's B1 and B2 on MuLMS: the encoder's gradient checkpointing")
    parser.add_argument('--max-train', type=int, default=0,
                        help='smoke test: the first N training and tune sentences only')
    parser.add_argument('--cache', default=None)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--out-dir', default=str(OUT_DIR))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--device', default=None)
    parser.add_argument('--threads', type=int, default=0)
    args = parser.parse_args(argv)
    if args.seed_offset and not args.tag:
        parser.error('--seed-offset needs --tag, so that offset runs get their own results file')
    if args.context < 0:
        parser.error('--context: 0 (off) or more sub-words')
    if args.span_head and args.joint_sofc:
        parser.error('--span-head and --joint-sofc each add a head: not both (P1)')
    if args.screen and args.save_predictions:
        parser.error('--screen has no test predictions to save')
    if (args.screen or any(getattr(args, k) for k in P1_OPTIONS)) and not args.tag:
        parser.error("P1's runs need a --tag (each cell writes its own results file)")
    return args


P1_OPTIONS = ('context', 'joint_sofc', 'span_head', 'gradient_checkpointing', 'screen')


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return run(parse_args(argv))


if __name__ == '__main__':
    main()
