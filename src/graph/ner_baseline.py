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
    negative log-likelihood and decoded with Viterbi).
  - Metrics: seqeval micro F1 in default (conlleval) mode (primary) and
    strict IOB2 mode, plus per-class scores. The 2024 paper does not state
    its averaging ("averaging over the five random initializations and
    entity types"); the authors' 2025 follow-up uses micro F1. Macro and
    support-weighted F1 are therefore reported as well: on this data macro F1
    is several points lower than micro F1.

Results are written to <PROCESSED_DIR>/ner/ner_results_<granularity>[_<protocol>].json
after every run, and completed runs are skipped when re-run; re-running a
finished protocol only refreshes its `summary` (mean, population SD and 95%
Student-t CI over runs for micro, strict micro, macro and weighted F1).

Training variants (ROADMAP P1) write their own files with a `_cw` or `_crf`
suffix; `--compare` pairs them with the baseline run by run (same split and
model seed) and writes ner_comparisons.json.

Usage:
  python ner_baseline.py --granularity fine --seeds 0 1 2 3 4
  python ner_baseline.py --granularity coarse --epochs 2 --max-train 32   # smoke test
  python ner_baseline.py --granularity fine --vary-split --class-weights
  python ner_baseline.py --compare                                        # paired comparison
"""

import argparse
import json
import os
import random
import re
import statistics
import time
from collections import Counter

import conll
import metrics
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


def train_one(seed, train, val, test, labels, args, device):
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
    return result


# F1 averages reported for every run (seqeval, in %): micro is the primary one
F1_AVERAGES = {
    'micro': lambda r: r['f1'],
    'micro_strict': lambda r: r['f1_strict'],
    'macro': lambda r: 100 * r['per_class']['macro avg']['f1-score'],
    'weighted': lambda r: 100 * r['per_class']['weighted avg']['f1-score'],
}

# Results files of training variants: ner_results_<granularity>[_<protocol>]_<variant>.json
VARIANT_FILE = re.compile(r'^ner_results_(fine|coarse)(_random_split_vary|_leave_one_paper_out)?'
                          r'_(cw|crf)\.json$')


def variant_name(args):
    """Training variant of a run: 'baseline', 'cw' (class weights) or 'crf'."""
    if getattr(args, 'crf', False):
        return 'crf'
    return 'cw' if args.class_weights else 'baseline'


def resumable_runs(path, labels, model, variant):
    """Runs of an earlier invocation that this one may reuse.

    Only runs with the same labels, model and training variant count; files
    written before variants existed are baseline runs.
    """
    if not os.path.exists(path):
        return []
    with open(path) as f:
        previous = json.load(f)
    same = (previous.get('labels') == labels and previous.get('model') == model
            and previous.get('variant', 'baseline') == variant)
    return previous.get('runs', []) if same else []


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
    for leave-one-paper-out), so both members of a pair share the split and the
    model seed. For every F1 average: the per-run differences, their mean and
    the 95% Student-t CI of the mean difference.
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
    """Pair every variant results file in `directory` with its baseline file.

    Writes <directory>/ner_comparisons.json and returns its content.
    """
    comparisons = {}
    for fn in sorted(os.listdir(directory)):
        m = VARIANT_FILE.match(fn)
        if not m:
            continue
        base_fn = f"ner_results_{m.group(1)}{m.group(2) or ''}.json"
        if not os.path.exists(os.path.join(directory, base_fn)):
            print(f"  {fn}: no baseline file {base_fn}, skipped")
            continue
        with open(os.path.join(directory, base_fn)) as f:
            base = json.load(f)
        with open(os.path.join(directory, fn)) as f:
            var = json.load(f)
        comp = compare_to_baseline(base, var)
        comparisons[fn[:-len('.json')]] = {
            'variant': m.group(3), 'granularity': m.group(1),
            'protocol': var.get('protocol'), 'baseline_file': base_fn, **comp}
        d = comp['micro']
        print(f"  {fn}: micro F1 difference {d['mean_diff']:+.2f}, 95% CI {d['ci95']} "
              f"over {comp['n_pairs']} paired runs")
    with open(os.path.join(directory, 'ner_comparisons.json'), 'w') as f:
        json.dump(comparisons, f, indent=2)
    return comparisons


def make_folds(data, args):
    """[(name, train, val, test)] for the chosen protocol."""
    if args.leave_one_paper_out:
        folds = []
        for doc in sorted({d for _, _, d in data}):
            rest = [x for x in data if x[2] != doc]
            order = list(range(len(rest)))
            random.Random(args.split_seed).shuffle(order)
            n_val = round(0.15 * len(rest))
            val = [rest[i] for i in order[:n_val]]
            train = [rest[i] for i in order[n_val:]]
            folds.append((doc, train, val, [x for x in data if x[2] == doc]))
        return folds
    return None


def run(args):
    import torch

    torch.set_num_threads(args.threads or os.cpu_count())
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.join(PROCESSED_DIR, 'ner')
    os.makedirs(out_dir, exist_ok=True)

    for granularity in (['fine', 'coarse'] if args.granularity == 'both'
                        else [args.granularity]):
        data = load_split_corpus(granularity)
        top_k = args.top_k if granularity == 'fine' else None
        labels, tie_info = select_labels(data, top_k)
        keep = set(labels)
        data = [(t, restrict_tags(g, keep), d) for t, g, d in data]
        protocol = 'leave_one_paper_out' if args.leave_one_paper_out else (
            'random_split_vary' if args.vary_split else 'random_split_fixed')
        variant = variant_name(args)
        suffix = '' if protocol == 'random_split_fixed' else f'_{protocol}'
        if variant != 'baseline':
            suffix += f'_{variant}'
        if args.max_train:
            suffix += f'_smoke{args.max_train}'
        out_path = os.path.join(out_dir, f'ner_results_{granularity}{suffix}.json')
        results = {'granularity': granularity, 'protocol': protocol, 'variant': variant,
                   'model': args.model, 'labels': labels, 'label_tie': tie_info,
                   'paper_test_f1': PAPER_F1[granularity],
                   'config': {k: v for k, v in vars(args).items()
                              if k not in ('granularity', 'compare')},
                   'runs': resumable_runs(out_path, labels, args.model, variant)}
        done = {r.get('run_id') for r in results['runs']}

        print(f"\n=== NER {granularity}: {len(data)} sentences, {len(labels)} classes, "
              f"protocol {protocol}, variant {variant}, device {device} ===")
        if tie_info:
            print(f"  label cut tie: {tie_info}")

        folds = make_folds(data, args)
        if folds is None:
            folds = []
            for seed in args.seeds:
                split_seed = seed if args.vary_split else args.split_seed
                tr, va, te = random_split(len(data), split_seed)
                folds.append((f'seed{seed}', [data[i] for i in tr],
                              [data[i] for i in va], [data[i] for i in te]))
        for run_id, train, val, test in folds:
            if run_id in done:
                print(f"  {run_id}: already done, skipping")
                continue
            if args.max_train:
                train = train[:args.max_train]
            seed = int(run_id[4:]) if run_id.startswith('seed') else args.seeds[0]
            start = time.time()
            res = train_one(seed, train, val, test, labels, args, device)
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


def main():
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
    parser.add_argument('--max-train', type=int, default=0,
                        help='limit training sentences (smoke tests)')
    parser.add_argument('--device', default=None)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--compare', nargs='?', const=os.path.join(PROCESSED_DIR, 'ner'),
                        default=None, metavar='DIR',
                        help='no training: pair every variant results file in DIR '
                             '(default: <PROCESSED_DIR>/ner) with its baseline and '
                             'write ner_comparisons.json')
    args = parser.parse_args()
    if args.compare:
        compare_all(args.compare)
        return
    run(args)


if __name__ == '__main__':
    main()
