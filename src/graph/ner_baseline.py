"""
ner_baseline.py - MatSciBERT NER baseline (MaterioMiner protocol)

The only published results on MaterioMiner are NER results (Kumar et al.,
Sci Data 2024): MatSciBERT fine-tuned on a random 65/15/20 sentence split,
five random initialisations, entity-level seqeval F1 (exact span and type):

    fine-grained (27 most frequent classes)   test F1 69.92
    coarse-grained (27 types)                 test F1 72.32

This script reproduces that protocol so the project has a number that is
comparable with the literature (the pipeline itself starts from gold spans).
The paper's exact split is not published; compare mean +/- std, not single
runs.

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
a run counts as done once both files hold it.

`--label-semantics` (with `--crf`; ROADMAP §4, week-3 protocol, variant 2)
replaces the linear token classifier by a label-semantic head: the score of B-X
(I-X) is the dot product of the token's state with W_B e_X (W_I e_X), e_X the
embedding of a description of X, and O has a learned vector (make_label_head).
The descriptions (ner_label_text.py: MMO definitions of the fine-grained
classes, the MMO classes annotated with each coarse-grained type) are embedded
in every run by the pre-trained encoder, before fine-tuning. Files `_crf_ls`,
compared with `crf`; the config stores the SHA-256 of the descriptions.

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
  python ner_baseline.py --compare                                        # paired comparison
"""

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from types import SimpleNamespace

import conll
import metrics
import ner_label_text
from config import PROCESSED_DIR, RAW_DIR

PAPER_F1 = {'fine': 69.92, 'coarse': 72.32}
DEFAULT_MODEL = 'm3rg-iitd/matscibert'


def load_split_corpus(granularity):
    """[(tokens, tags, doc_id)] for one granularity (files read separately)."""
    folder = 'fine_grained_ner' if granularity == 'fine' else 'coarse_grained_ner'
    datadir = os.path.join(RAW_DIR, folder)
    data = []
    for fn in sorted(f for f in os.listdir(datadir) if f.endswith('.conll')):
        for tokens, tags in conll.read_conll_sentences(os.path.join(datadir, fn)):
            data.append((tokens, tags, fn))
    return data


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


def encode(tokenizer, sentences, label2id, max_length=512):
    """Tokenise pre-split words; label the first subword of each word."""
    features = []
    for tokens, tags in sentences:
        enc = tokenizer(tokens, is_split_into_words=True, truncation=False)
        if len(enc['input_ids']) > max_length:
            raise ValueError(f"Sentence of {len(tokens)} words needs "
                             f"{len(enc['input_ids'])} subwords (> {max_length})")
        labels = []
        prev_word = None
        for word_id in enc.word_ids():
            if word_id is None or word_id == prev_word:
                labels.append(-100)
            else:
                labels.append(label2id[tags[word_id]])
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

        def nll(self, emissions, tags, mask):
            """Mean negative log-likelihood of the gold tag sequences.

            emissions (B, W, L), tags (B, W), mask (B, W): True on words,
            left-aligned, at least one word per sequence.
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
            return (log_z - gold).mean()

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


def train_one(seed, train, val, test, labels, args, device, test_index=None, descriptions=None):
    """Fine-tune and test once; `seed` is the model seed (--seed-offset included).

    test_index (positions of `test` in the corpus) is stored with the
    predictions when --save-predictions is set. descriptions (--label-semantics):
    a text per label of `labels`; the token classifier is then a label-semantic
    head over their embeddings by the pre-trained encoder (make_label_head).
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
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(label_list), id2label=id2label, label2id=label2id
    ).to(device)
    if descriptions is not None:
        # embedded before any fine-tuning; the head replaces the linear classifier
        assert len(descriptions) == len(labels), 'one description per label'
        embeddings = embed_descriptions(model.base_model, tokenizer, descriptions, device)
        model.classifier = make_label_head(embeddings, model.config).to(device)
    pad_id = tokenizer.pad_token_id
    crf = make_crf(label_list).to(device) if getattr(args, 'crf', False) else None

    f_train = encode(tokenizer, [(t, g) for t, g, _ in train], label2id)
    f_val = encode(tokenizer, [(t, g) for t, g, _ in val], label2id)
    f_test = encode(tokenizer, [(t, g) for t, g, _ in test], label2id)

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
        for ids, mask, lab, _ in batches(f_train, args.batch_size, pad_id,
                                         shuffle_seed=seed * 1000 + epoch):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            if crf is None:
                loss = loss_fn(logits.view(-1, logits.size(-1)), lab.to(device).view(-1))
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
    test_pred = predict(model, f_test, args.batch_size, pad_id, id2label, device, crf)
    result = score([g for _, g, _ in test], test_pred)
    result.update({'seed': seed, 'best_epoch': best[2], 'val_f1': best[0],
                   'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
                   'history': history})
    if getattr(args, 'save_predictions', False):
        result.update(prediction_fields(test_index, test, test_pred))
    return result


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


def encode_multitask(tokenizer, examples, label2id):
    """encode() for every head: {head: features}, in the order of `examples`.

    An example without tags for a head has only -100 labels for it, so that
    multitask_loss never passes it to that head's CRF.
    """
    features = {}
    for head in MT_HEADS:
        sentences = [(tokens, ['O'] * len(tokens) if tags[head] is None else tags[head])
                     for tokens, tags in examples]
        features[head] = encode(tokenizer, sentences, label2id[head])
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


def multitask_loss(logits, labels, crfs):
    """{head: mean CRF NLL over the rows of the batch that are labelled for the head}.

    logits {head: (B, T, L)} and labels {head: (B, T)} as from multitask_batches.
    Rows without labels for a head are dropped before its CRF, and a head with no
    labelled row in the batch is left out: it contributes nothing to the loss.
    """
    losses = {}
    for head, lab in labels.items():
        rows = (lab != -100).any(1)
        if rows.any():
            emissions = logits[head][rows.to(logits[head].device)]
            losses[head] = crfs[head].nll(*word_view(emissions, lab[rows]))
    return losses


class HeadView:
    """One head of a multi-task tagger, called like a token-classification model (predict)."""

    def __init__(self, tagger, head):
        self.tagger, self.head = tagger, head

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
            self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
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


def train_multitask(seed, parts, labels, args, device, test_index=None):
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
    other_best_epoch (the best epoch of the other head).
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
    pad_id = tokenizer.pad_token_id

    examples = multitask_examples(parts['fine'][0], parts['coarse'][0])
    f_train = encode_multitask(tokenizer, examples, label2id)
    f_val, f_test = ({h: encode(tokenizer, [(t, g) for t, g, _ in parts[h][k]], label2id[h])
                      for h in MT_HEADS} for k in (1, 2))

    optimizer = torch.optim.AdamW(
        [{'params': [*model.encoder.parameters(), *model.heads.parameters()]},
         {'params': model.crfs.parameters(), 'lr': CRF_LR, 'weight_decay': 0.0}],
        lr=args.lr, weight_decay=0.01)
    params = list(model.parameters())
    steps = args.epochs * ((len(examples) + args.batch_size - 1) // args.batch_size)
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
        for ids, mask, lab, _ in multitask_batches(f_train, args.batch_size, pad_id,
                                                    shuffle_seed=seed * 1000 + epoch):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device))
            losses = multitask_loss(logits, lab, model.crfs)
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
    for h in MT_HEADS:
        model.load_state_dict(best[h][1])
        train, val, test = parts[h]
        test_pred = predict(HeadView(model, h), f_test[h], args.batch_size, pad_id, id2label[h],
                            device, model.crfs[h])
        result = score([g for _, g, _ in test], test_pred)
        other = [o for o in MT_HEADS if o != h][0]
        result.update({'seed': seed, 'best_epoch': best[h][2], 'val_f1': best[h][0],
                       'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
                       'history': multitask_history(log, h),
                       'multi_task': {'n_examples': len(examples), 'n_split_examples': n_split,
                                      'other_best_epoch': best[other][2]}})
        if getattr(args, 'save_predictions', False):
            result.update(prediction_fields(test_index, test, test_pred))
        records[h] = result
    return records


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
RESULT_FILE = re.compile(r'ner_results_(fine|coarse)(?:_(random_split_vary|leave_one_paper_out))?'
                         r'(?:_([a-z0-9_]+))?\.json')
# untagged variants: a tagged variant must not take the name of one (--crf --tag mt: crf_mt)
PLAIN_VARIANTS = ('baseline', 'cw', 'crf', 'crf_mt', 'crf_ls')


def variant_name(args):
    """Training variant of a run: 'baseline', 'cw', 'crf', 'crf_mt' or 'crf_ls', then the tag.

    'cw' is --class-weights, 'crf_mt' is --crf --multi-task and 'crf_ls' is --crf
    --label-semantics. A tag is appended with an underscore (crf_m100); the baseline
    is named by it alone.
    """
    if getattr(args, 'crf', False):
        name = 'crf_mt' if getattr(args, 'multi_task', False) else 'crf'
        if getattr(args, 'label_semantics', False):
            name = 'crf_ls'
    else:
        name = 'cw' if args.class_weights else 'baseline'
    tag = getattr(args, 'tag', None)
    if not tag:
        return name
    return tag if name == 'baseline' else f'{name}_{tag}'


def reference_variant(variant):
    """Variant that a variant is compared with ('baseline': the baseline file).

    crf_mt[_<tag>], crf_ls[_<tag>] and crf_<tag> are compared with crf and cw_<tag>
    with cw; cw, crf and a bare tag with the baseline.
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

    Files written before variants existed hold baseline runs, and files without
    a stored offset have offset 0.
    """
    return {'labels': results.get('labels'), 'model': results.get('model'),
            'variant': results.get('variant', 'baseline'),
            'seed_offset': results.get('config', {}).get('seed_offset', 0)}


def resumable_runs(path, labels, model, variant, seed_offset=0, save_predictions=False):
    """Runs of an earlier invocation that this one may reuse.

    Only runs with the same labels, model, training variant and seed offset
    count (see stored_settings). With `save_predictions`, runs that did not
    store their predictions are not reused.
    """
    if not os.path.exists(path):
        return []
    with open(path) as f:
        previous = json.load(f)
    same = stored_settings(previous) == {'labels': labels, 'model': model, 'variant': variant,
                                         'seed_offset': seed_offset}
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


def check_overwrite(path, labels, model, variant, seed_offset=0, save_predictions=False):
    """Raise OverwriteError if `path` holds runs that resumable_runs would not reuse.

    A run writes its results file anew, so those runs would be lost: runs made
    with other labels, model, variant or seed offset, and, with
    `save_predictions`, runs that did not store their predictions.
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        previous = json.load(f)
    held = previous.get('runs', [])
    kept = resumable_runs(path, labels, model, variant, seed_offset, save_predictions)
    if len(kept) >= len(held):
        return
    stored = stored_settings(previous)
    now = {'labels': labels, 'model': model, 'variant': variant, 'seed_offset': seed_offset}
    reasons = [setting_change(k, stored[k], now[k]) for k in stored if stored[k] != now[k]]
    if not reasons:
        missing = [r.get('run_id') for r in held if 'test_pred' not in r]
        reasons = [f'--save-predictions is set and runs {missing} have no stored predictions']
    raise OverwriteError(
        f"{path}: runs that this invocation cannot resume ({'; '.join(reasons)}) would be "
        f"replaced ({len(held) - len(kept)} of {len(held)} stored). Use --tag NAME to write "
        f"the new runs to a file of their own, or --overwrite to replace the old runs on purpose")


def check_results_files(args, files):
    """check_overwrite for every (path, labels) that a run will write.

    Nothing is checked with --overwrite, or for smoke-test files (--max-train),
    which may be replaced at will.
    """
    if args.max_train or getattr(args, 'overwrite', False):
        return
    for path, labels in files:
        check_overwrite(path, labels, args.model, variant_name(args), args.seed_offset,
                        args.save_predictions)


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


def make_folds(data, args, indices=False):
    """[(name, train, val, test)] for leave-one-paper-out; None for random splits.

    The parts are lists of data tuples, or of their positions in `data` when
    `indices` is set (the test positions go with the stored predictions).
    """
    if args.leave_one_paper_out:
        folds = []
        for doc in sorted({d for _, _, d in data}):
            rest = [i for i, x in enumerate(data) if x[2] != doc]
            order = list(range(len(rest)))
            random.Random(args.split_seed).shuffle(order)
            n_val = round(0.15 * len(rest))
            val = [rest[i] for i in order[:n_val]]
            train = [rest[i] for i in order[n_val:]]
            test = [i for i, x in enumerate(data) if x[2] == doc]
            if not indices:
                train, val, test = ([data[i] for i in part] for part in (train, val, test))
            folds.append((doc, train, val, test))
        return folds
    return None


def granularity_data(granularity, args):
    """(data, labels, tie info) of a granularity, tags of the classes not kept mapped to O.

    Fine: the --top-k classes; coarse: all types.
    """
    data = load_split_corpus(granularity)
    labels, tie_info = select_labels(data, args.top_k if granularity == 'fine' else None)
    keep = set(labels)
    return [(t, restrict_tags(g, keep), d) for t, g, d in data], labels, tie_info


def protocol_name(args):
    return 'leave_one_paper_out' if args.leave_one_paper_out else (
        'random_split_vary' if args.vary_split else 'random_split_fixed')


def results_path(out_dir, granularity, args):
    """<out_dir>/ner_results_<granularity>[_<protocol>][_<variant>][_smoke<N>].json"""
    suffix = result_suffix(protocol_name(args), variant_name(args))
    if args.max_train:
        suffix += f'_smoke{args.max_train}'
    return os.path.join(out_dir, f'ner_results_{granularity}{suffix}.json')


def stored_config(args):
    """The arguments saved as `config` (the granularity is a field of its own)."""
    return {k: v for k, v in vars(args).items()
            if k not in ('granularity', 'compare', 'overwrite')}


def torch_device(args):
    """Set the number of torch threads and return the device to train on."""
    import torch
    torch.set_num_threads(args.threads or os.cpu_count())
    return args.device or ('cuda' if torch.cuda.is_available() else 'cpu')


def run(args):
    out_dir = os.path.join(PROCESSED_DIR, 'ner')
    if getattr(args, 'multi_task', False):
        run_multitask(args, out_dir)
        return
    granularities = ['fine', 'coarse'] if args.granularity == 'both' else [args.granularity]
    tasks = {g: granularity_data(g, args) for g in granularities}
    semantics = getattr(args, 'label_semantics', False)
    descriptions = {g: label_descriptions(g, tasks[g][1]) if semantics else None
                    for g in granularities}
    # every results file is checked before any training, so a run never stops halfway
    check_results_files(args, [(results_path(out_dir, g, args), tasks[g][1])
                               for g in granularities])
    device = torch_device(args)
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
                                          args.seed_offset, args.save_predictions)}
        if semantics:
            results['config']['label_descriptions_sha256'] = ner_label_text.descriptions_sha256(
                labels, descriptions[granularity])
        done = {r.get('run_id') for r in results['runs']}

        offset = f", seed offset {args.seed_offset}" if args.seed_offset else ''
        print(f"\n=== NER {granularity}: {len(data)} sentences, {len(labels)} classes, "
              f"protocol {protocol}, variant {variant}{offset}, device {device} ===")
        if tie_info:
            print(f"  label cut tie: {tie_info}")
        if semantics:
            print(f"  label descriptions: sha256 "
                  f"{results['config']['label_descriptions_sha256'][:12]}")

        folds = make_folds(data, args, indices=True)     # parts as positions in `data`
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
            seed += args.seed_offset      # the model seed moves, the split and run_id do not
            start = time.time()
            res = train_one(seed, train, val, test, labels, args, device, test_index=te,
                            descriptions=descriptions[granularity])
            res['run_id'] = run_id
            res['runtime_sec'] = round(time.time() - start, 1)
            results['runs'].append(res)
            summarize(results)
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)
            # the published numbers are in-distribution; no reference for held-out papers
            ref = ('' if args.leave_one_paper_out
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
    data, labels, ties = {}, {}, {}
    for head in MT_HEADS:
        data[head], labels[head], ties[head] = granularity_data(head, args)
    check_aligned(data['fine'], data['coarse'])
    paths = {head: results_path(out_dir, head, args) for head in MT_HEADS}
    check_results_files(args, [(paths[head], labels[head]) for head in MT_HEADS])
    device = torch_device(args)
    os.makedirs(out_dir, exist_ok=True)
    protocol, variant, config = protocol_name(args), variant_name(args), stored_config(args)
    results = {head: {'granularity': head, 'protocol': protocol, 'variant': variant,
                      'model': args.model, 'labels': labels[head], 'label_tie': ties[head],
                      'paper_test_f1': PAPER_F1[head], 'config': config,
                      'runs': resumable_runs(paths[head], labels[head], args.model, variant,
                                             args.seed_offset, args.save_predictions)}
               for head in MT_HEADS}
    done = set.intersection(*({r.get('run_id') for r in results[h]['runs']} for h in MT_HEADS))

    def save():
        for head in MT_HEADS:
            summarize(results[head])
            with open(paths[head], 'w') as f:
                json.dump(results[head], f, indent=2)

    offset = f", seed offset {args.seed_offset}" if args.seed_offset else ''
    print(f"\n=== NER multi-task: {len(data['fine'])} sentences, {len(labels['fine'])} fine and "
          f"{len(labels['coarse'])} coarse classes, protocol {protocol}, variant {variant}"
          f"{offset}, device {device} ===")
    for head in MT_HEADS:
        if ties[head]:
            print(f"  {head} label cut tie: {ties[head]}")

    folds = make_folds(data['fine'], args, indices=True)     # parts as corpus positions
    if folds is None:
        folds = []
        for seed in args.seeds:
            split_seed = seed if args.vary_split else args.split_seed
            folds.append((f'seed{seed}', *random_split(len(data['fine']), split_seed)))
    # the published numbers are in-distribution; no reference for held-out papers
    ref = {h: '' if args.leave_one_paper_out else f" (paper {PAPER_F1[h]})" for h in MT_HEADS}
    for run_id, tr, va, te in folds:
        if run_id in done:
            print(f"  {run_id}: already done, skipping")
            continue
        if args.max_train:
            tr = tr[:args.max_train]
        parts = {h: tuple([data[h][i] for i in part] for part in (tr, va, te)) for h in MT_HEADS}
        seed = int(run_id[4:]) if run_id.startswith('seed') else args.seeds[0]
        seed += args.seed_offset
        start = time.time()
        records = train_multitask(seed, parts, labels, args, device, test_index=te)
        runtime = round(time.time() - start, 1)
        for head, res in records.items():
            res['run_id'] = run_id
            res['runtime_sec'] = runtime      # the shared training time
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


def tag_name(value):
    """argparse type of --tag: lowercase letters and digits (the tag goes into file names)."""
    if not re.fullmatch(r'[a-z0-9]+', value):
        raise argparse.ArgumentTypeError(f"invalid tag {value!r}: use lowercase letters and digits")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='MatSciBERT NER baseline')
    parser.add_argument('--granularity', choices=['fine', 'coarse', 'both'], default='both')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--vary-split', action='store_true',
                        help='new random split per seed')
    parser.add_argument('--leave-one-paper-out', action='store_true')
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
                             'tagged _crf_ls)')
    parser.add_argument('--tag', type=tag_name, default=None, metavar='NAME',
                        help='appended to the variant name and the results file (crf + m100: '
                             'crf_m100; baseline + m100: m100); lowercase letters and digits')
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
    if getattr(args, 'label_semantics', False):
        if not args.crf:
            parser.error('--label-semantics needs --crf (its scores are the CRF emissions)')
        if getattr(args, 'multi_task', False):
            parser.error('--label-semantics is not available with --multi-task')
    if getattr(args, 'multi_task', False):
        if not args.crf:
            parser.error('--multi-task needs --crf (a CRF head per granularity)')
        if args.granularity != 'both':
            parser.error('--multi-task trains the fine and the coarse head together: '
                         '--granularity must be both (the default)')
    if args.tag and variant_name(args) in PLAIN_VARIANTS:
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
