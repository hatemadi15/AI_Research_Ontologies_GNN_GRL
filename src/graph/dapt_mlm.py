"""
dapt_mlm.py - domain-adaptive pre-training of MatSciBERT on PubMed (masked LM)

Continued masked-language-model training of MatSciBERT on the PubMed fatigue
sentences in data/raw/dataset/pubmed_sentences.txt (week-2 protocol in
docs/ROADMAP.md, P1 item 5). The saved model is then fine-tuned with
`ner_baseline.py --crf --model <out>`. The settings are pre-registered: the
flags of the fixed settings exist for smoke tests, not for tuning.

  - Corpus: one sentence per line, blank lines skipped. Exact duplicates are
    dropped (first occurrence kept), then every sentence that shares a word
    8-gram with MaterioMiner text: every sentence of the fine- and the
    coarse-grained CoNLL files (they tokenise a few sentences differently, so
    both versions count), tokens joined by spaces.
  - Normalisation for both tests: Unicode NFKC, lowercase, every dash
    (category Pd) and the minus sign U+2212 -> '-', then the tokens are the
    matches of \\w+|[^\\w\\s]: runs of letters, digits and '_', or one other
    non-space character, so punctuation is a token of its own and
    "Low-cycle fatigue (LCF)" equals "low - cycle fatigue ( LCF )". Duplicates
    have equal token lists; an overlap is any shared run of 8 tokens.
  - Held-out set: random.Random(seed) shuffles the kept sentences and the
    first round(5%) are held out. They only serve to report MLM loss and
    perplexity, with one fixed set of masks (seeded from (seed, 'heldout')).
  - Tokenisation: MatSciBERT tokenizer, truncation at 128 tokens, dynamic
    padding. Batches are grouped by length: shuffle, sort within chunks of 50
    batches, shuffle the batch order, all from (seed, epoch).
  - Masking (BERT): every token except the special ones ([CLS], [SEP],
    [PAD], [UNK], [MASK]) is selected with probability 0.15; a selected token
    becomes [MASK] (80%), a token drawn uniformly from the vocabulary (10%)
    or stays unchanged (10%). The masks of a step come from a torch.Generator
    seeded from (seed, epoch, step), never from global RNG state. Dropout uses
    the global RNG, so it is reseeded from (seed, epoch, step) before every step.
  - Optimisation: AdamW, lr 5e-5, weight decay 0.01 (none for biases and
    LayerNorm weights), linear warm-up over ceil(6%) of all steps then linear
    decay to 0, batch 32, gradient-norm clipping at 1.0, 3 epochs, seed 0
    (transformers.set_seed). The loss is the mean cross-entropy over the
    masked tokens, as in BertForMaskedLM; the prediction head only runs at the
    masked positions (same loss, about 20% faster on CPU).

Resuming. Every --save-every steps and at each epoch end the training state
(model, optimiser, scheduler, epoch, step in the epoch, loss log, held-out
results) is written atomically to <out>/state/state.pt. Re-running the same
command resumes from it with the same batch order, masks and dropout, so a run
split across jobs equals an uninterrupted one up to floating-point
nondeterminism. A state written with other settings or data (a hash of both is
stored) is refused. --time-budget-min M checkpoints and exits 0 once the next
checkpoint would end after M minutes (each job makes at least one step).
--max-steps N stops the same way after N steps in total; the schedule still
spans all epochs. The model and tokenizer are written to --out only when all
epochs are done; the state then keeps only the logs. The report (--report)
holds the corpus counts, settings, step counts, held-out MLM loss and
perplexity before training and after each epoch, train loss per epoch,
runtime and library versions.

Usage:
  python src/graph/dapt_mlm.py --threads 2 --time-budget-min 110     # repeat until done
  python src/graph/dapt_mlm.py --max-sentences 640 --epochs 1 --save-every 5 \\
      --out /tmp/dapt_smoke --report /tmp/dapt_smoke.json              # smoke test
  python src/graph/ner_baseline.py --crf --vary-split --tag dapt \\
      --model data/processed/dapt/matscibert_pubmed
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import time
import unicodedata

import conll
from config import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR

DEFAULT_MODEL = 'm3rg-iitd/matscibert'
DEFAULT_CORPUS = os.path.join(RAW_DIR, 'pubmed_sentences.txt')
DEFAULT_OUT = os.path.join(PROCESSED_DIR, 'dapt', 'matscibert_pubmed')
DEFAULT_REPORT = os.path.join(PROJECT_ROOT, 'results', 'dapt_mlm.json')
NER_DIRS = ('fine_grained_ner', 'coarse_grained_ner')
STATE_FILE = 'state.pt'
STATE_FORMAT = 1        # bump when the training procedure changes
EVAL_BATCH_SIZE = 64
N_EXAMPLES = 20         # dropped sentences listed in the report
WEIGHT_KEYS = ('model', 'optimizer', 'scheduler')

# Arguments that determine the training trajectory; hashed into the state with the data
TRAJECTORY_ARGS = ('model', 'max_sentences', 'ngram', 'heldout_frac', 'seed', 'epochs',
                   'batch_size', 'lr', 'weight_decay', 'warmup_ratio', 'max_grad_norm',
                   'max_length', 'mlm_prob', 'chunk_batches')

NORMALISATION = ("NFKC; lowercase; dashes (Unicode Pd) and U+2212 -> '-'; tokens = regex "
                 "\\w+|[^\\w\\s]. Duplicates: equal token lists (first occurrence kept). "
                 "Overlap: a shared run of n tokens with any MaterioMiner sentence (fine- and "
                 "coarse-grained CoNLL files, tokens joined by spaces).")
TOKEN_RE = re.compile(r'\w+|[^\w\s]')


# ---- corpus: leakage filter and held-out split (standard library only) ----

def normalise(text):
    """Tokens for the duplicate and overlap tests (see NORMALISATION)."""
    text = unicodedata.normalize('NFKC', text).lower()
    text = ''.join('-' if ch == '\u2212' or unicodedata.category(ch) == 'Pd' else ch
                   for ch in text)
    return TOKEN_RE.findall(text)


def ngrams(tokens, n):
    """Token n-grams in order (none for fewer than n tokens)."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def filter_corpus(sentences, reference, n=8):
    """Drop duplicates, then sentences sharing a token n-gram with `reference`.

    sentences: list of str; reference: (doc, sentence) pairs. The first
    occurrence of a duplicate is kept and goes on to the overlap test.
    Returns (kept, dropped); dropped holds 'duplicate' ({index, sentence,
    duplicate_of}) and 'overlap' ({index, sentence, ngram, docs}) lists, with
    indices into `sentences`, and the size of the reference.
    """
    index = {}
    for doc, text in reference:
        for gram in ngrams(normalise(text), n):
            index.setdefault(gram, set()).add(doc)
    first = {}
    kept, duplicate, overlap = [], [], []
    for i, text in enumerate(sentences):
        tokens = tuple(normalise(text))
        if tokens in first:
            duplicate.append({'index': i, 'sentence': text, 'duplicate_of': first[tokens]})
            continue
        first[tokens] = i
        hit = next((g for g in ngrams(tokens, n) if g in index), None)
        if hit is None:
            kept.append(text)
        else:
            overlap.append({'index': i, 'sentence': text, 'ngram': ' '.join(hit),
                            'docs': sorted(index[hit])})
    return kept, {'duplicate': duplicate, 'overlap': overlap,
                  'reference_sentences': len(reference), 'reference_ngrams': len(index)}


def read_sentences(path):
    """Stripped non-blank lines of a one-sentence-per-line file, and the blank-line count."""
    with open(path, encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    sentences = [s for s in lines if s]
    return sentences, len(lines) - len(sentences)


def materiominer_sentences(raw_dir=RAW_DIR):
    """(file name, sentence) for every sentence of the fine- and coarse-grained files."""
    out = []
    for sub in NER_DIRS:
        folder = os.path.join(raw_dir, sub)
        for fn in sorted(f for f in os.listdir(folder) if f.endswith('.conll')):
            out += [(fn, ' '.join(tokens))
                    for tokens, _ in conll.read_conll_sentences(os.path.join(folder, fn))]
    return out


def split_heldout(sentences, frac, seed):
    """(train, heldout): random.Random(seed) shuffles, the first round(frac * n) are held out."""
    order = list(range(len(sentences)))
    random.Random(seed).shuffle(order)
    n_held = round(frac * len(sentences))
    return [sentences[i] for i in order[n_held:]], [sentences[i] for i in order[:n_held]]


def derive_seed(*parts):
    """63-bit seed from e.g. (seed, 'mask', epoch, step); the same on every run and machine."""
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:8], 'little') >> 1


def length_grouped_batches(lengths, batch_size, seed, epoch, chunk_batches=50):
    """Index batches of one epoch, grouped by length to cut padding.

    Shuffle all indices, sort each chunk of chunk_batches * batch_size by
    length (stable), cut it into batches, then shuffle the batch order. Only
    the last batch of the last chunk can be short. Deterministic given
    (seed, epoch).
    """
    rng = random.Random(derive_seed(seed, 'batches', epoch))
    order = list(range(len(lengths)))
    rng.shuffle(order)
    size = chunk_batches * batch_size
    batches = []
    for start in range(0, len(order), size):
        chunk = sorted(order[start:start + size], key=lambda i: lengths[i])
        batches += [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return batches


def settings_hash(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]


def check_resumable(state, settings, digest, path):
    """Refuse a training state written with other settings or data."""
    if state.get('settings_hash') == digest:
        return
    old = state.get('settings', {})
    diff = sorted(k for k in set(old) | set(settings) if old.get(k) != settings.get(k))
    raise SystemExit(f"{path} was written with other settings or data "
                     f"(differ: {', '.join(diff) or 'hash'}); use another --out, or delete "
                     f"that state to start over")


def next_checkpoint_late(elapsed, budget, steps_sec, save_sec, eval_sec=0.0):
    """True when the steps up to the next checkpoint, the evaluation (at an epoch
    end) and the save would end after the budget (all in seconds)."""
    return budget > 0 and elapsed + steps_sec + eval_sec + save_sec > budget


def stop_reason(args, gstep, step, padded, job_steps, sec_per_token, timing, elapsed):
    """Why training stops before this step (--max-steps, --time-budget-min), or None.

    padded: padded tokens of each batch of the epoch. The next checkpoint is
    the next multiple of --save-every or the epoch end (which adds the held-out
    evaluation); the steps up to it are timed by their padded tokens, since
    step times vary with batch length. The budget is only checked after this
    job's first step, so every job makes progress.
    """
    if args.max_steps and gstep >= args.max_steps:
        return f"Reached --max-steps {args.max_steps}."
    if args.time_budget_min > 0 and job_steps > 0:
        ahead = min(args.save_every - gstep % args.save_every, len(padded) - step)
        eval_sec = timing.get('eval', 0.0) if ahead == len(padded) - step else 0.0
        steps_sec = sec_per_token * sum(padded[step:step + ahead])
        if next_checkpoint_late(elapsed, 60 * args.time_budget_min, steps_sec,
                                timing.get('save', 0.0), eval_sec):
            return (f"Time budget: the next checkpoint (global step {gstep + ahead}) would "
                    f"end after {args.time_budget_min:g} min.")
    return None


# ---- tensors: masking, batches, loss (torch imported lazily) ----

def mask_tokens(ids, special, generator, mask_id, vocab_size, mlm_prob=0.15):
    """BERT masking of a padded batch; returns (inputs, labels).

    ids (B, L) long; special (B, L) bool, True where nothing may be masked
    (special tokens, padding). Each other token is selected with probability
    mlm_prob and becomes mask_id (80%), a uniform random id (10%) or stays
    (10%). labels are -100 except at selected positions. All randomness comes
    from `generator`, with a fixed number of draws per call.
    """
    import torch
    select = torch.rand(ids.shape, generator=generator)
    action = torch.rand(ids.shape, generator=generator)
    random_ids = torch.randint(vocab_size, ids.shape, generator=generator)
    masked = (select < mlm_prob) & ~special
    labels = ids.masked_fill(~masked, -100)
    inputs = ids.clone()
    inputs[masked & (action < 0.8)] = mask_id
    replace = masked & (action >= 0.8) & (action < 0.9)
    inputs[replace] = random_ids[replace]
    return inputs, labels


def collate(seqs, pad):
    """Right-pad id lists to the longest: (ids, attention_mask) long tensors."""
    import torch
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    attention = torch.zeros((len(seqs), width), dtype=torch.long)
    for row, s in enumerate(seqs):
        ids[row, :len(s)] = torch.tensor(s, dtype=torch.long)
        attention[row, :len(s)] = 1
    return ids, attention


def special_mask(ids, attention, special_ids):
    import torch
    return torch.isin(ids, torch.as_tensor(special_ids)) | (attention == 0)


def heldout_examples(seqs, special_ids, pad_id, mask_id, vocab_size, seed, mlm_prob):
    """One fixed set of masks for the held-out sentences: [(inputs, labels)] id lists."""
    import torch
    if not seqs:
        return []
    ids, attention = collate(seqs, pad_id)
    gen = torch.Generator().manual_seed(derive_seed(seed, 'heldout'))
    inputs, labels = mask_tokens(ids, special_mask(ids, attention, special_ids), gen,
                                 mask_id, vocab_size, mlm_prob)
    return [(inputs[i, :len(s)].tolist(), labels[i, :len(s)].tolist())
            for i, s in enumerate(seqs)]


def mlm_loss(model, inputs, attention, labels, reduction='mean'):
    """Cross-entropy over the positions with labels != -100, as in BertForMaskedLM.

    For BERT the prediction head only runs at those positions.
    """
    import torch
    keep = labels != -100
    if hasattr(model, 'bert') and hasattr(model, 'cls'):
        hidden = model.bert(input_ids=inputs, attention_mask=attention).last_hidden_state
        logits = model.cls(hidden[keep])
    else:
        logits = model(input_ids=inputs, attention_mask=attention).logits[keep]
    return torch.nn.functional.cross_entropy(logits, labels[keep], reduction=reduction)


def evaluate(model, examples, pad_id, batch_size=EVAL_BATCH_SIZE):
    """Held-out MLM loss (mean over masked tokens) and perplexity, in eval mode."""
    import torch
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            chunk = [examples[i] for i in order[start:start + batch_size]]
            inputs, attention = collate([x for x, _ in chunk], pad_id)
            labels, _ = collate([y for _, y in chunk], -100)
            if (labels != -100).any():
                total += mlm_loss(model, inputs, attention, labels, reduction='sum').item()
                count += int((labels != -100).sum())
    model.train()
    loss = total / count if count else float('nan')
    return {'loss': loss, 'perplexity': math.exp(loss), 'masked_tokens': count}


def param_groups(model, weight_decay):
    """AdamW groups: weight decay for all but biases and LayerNorm weights."""
    import torch
    norm = {id(p) for m in model.modules() if isinstance(m, torch.nn.LayerNorm)
            for p in m.parameters(recurse=False)}
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if name.endswith('bias') or id(p) in norm else decay).append(p)
    return [{'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}]


def save_state(path, state):
    """Atomic torch.save: write <path>.tmp, fsync, rename over <path>."""
    import torch
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_state(path):
    """The saved training state, or None."""
    if not os.path.exists(path):
        return None
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)


def library_versions():
    import torch
    import transformers
    # str(): torch.__version__ is a TorchVersion, which weights-only loading refuses
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'transformers': str(transformers.__version__)}


def job_summary(job, times, padded, epoch_padded, n_batches, epochs):
    """Speed of this job and the projected time per epoch and for all epochs.

    Batch lengths vary a lot, so the projection scales the time per padded
    token (first step excluded: warm-up) to the padded tokens of an epoch.
    """
    if not times:
        return
    n = len(times)
    k = 1 if n > 1 else 0
    per_token = sum(times[k:]) / sum(padded[k:])
    epoch_sec = per_token * epoch_padded
    print(f"This job: {n} steps with {job['threads']} threads, {statistics.mean(times):.2f} s/step "
          f"(median {statistics.median(times):.2f}), {job['tokens'] / n:.0f} tokens/step "
          f"({job['padded_tokens'] / n:.0f} padded; the epoch averages "
          f"{epoch_padded / n_batches:.0f}). Projected from {1000 * per_token:.2f} ms per padded "
          f"token: {epoch_sec / 60:.0f} min per epoch of {n_batches} steps, "
          f"{epochs * epoch_sec / 3600:.2f} h for {epochs} epochs, plus held-out evaluations "
          f"and checkpoints.", flush=True)


def train(model, data, args, meta, state, state_path, t0):
    """Train from scratch (state None) or resume from `state`; returns (state, done).

    data: 'train' (token-id lists), 'heldout' ((inputs, labels) pairs with
    fixed masks), 'special_ids', 'pad_id', 'mask_id', 'vocab_size'. meta: the
    entries of a new state (settings and their hash, step counts, loading info).
    t0: time.perf_counter() at the start of the job (for --time-budget-min).
    """
    import torch
    from transformers import get_linear_schedule_with_warmup

    lengths = [len(s) for s in data['train']]
    n_batches = math.ceil(len(lengths) / args.batch_size)
    total = args.epochs * n_batches
    groups = param_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, math.ceil(args.warmup_ratio * total), total)
    job = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'threads': torch.get_num_threads(),
           'start_step': 0, 'end_step': 0, 'steps': 0, 'step_seconds': 0.0, 'tokens': 0,
           'padded_tokens': 0, 'eval_seconds': 0.0, 'save_seconds': 0.0, 'wall_seconds': 0.0,
           'versions': library_versions()}
    step_times, step_padded, window = [], [], []
    epoch_padded = 0

    def evaluate_heldout(epoch):
        t = time.perf_counter()
        res = evaluate(model, data['heldout'], data['pad_id'])
        res.update(epoch=epoch, seconds=round(time.perf_counter() - t, 1))
        state['heldout'].append(res)
        state['timing']['eval'] = res['seconds']
        job['eval_seconds'] += res['seconds']
        when = 'before training' if epoch == 0 else f'after epoch {epoch}'
        print(f"  held-out {when}: MLM loss {res['loss']:.4f}, perplexity {res['perplexity']:.3f} "
              f"({res['masked_tokens']:,} masked tokens, {res['seconds']:.0f} s)", flush=True)

    def checkpoint(gstep):
        t = time.perf_counter()
        job.update(end_step=gstep, wall_seconds=round(t - t0, 1))
        state.update(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict())
        save_state(state_path, state)
        for key in WEIGHT_KEYS:
            del state[key]
        sec = time.perf_counter() - t
        job['save_seconds'] += sec
        state['timing']['save'] = sec
        print(f"  saved the training state at global step {gstep} ({sec:.1f} s)", flush=True)

    print(f"AdamW: {sum(p.numel() for p in groups[0]['params']):,} parameters with weight decay, "
          f"{sum(p.numel() for p in groups[1]['params']):,} without; {n_batches} steps per epoch, "
          f"{total} in total, {math.ceil(args.warmup_ratio * total)} warm-up", flush=True)
    resumed = state is not None
    if resumed:
        model.load_state_dict(state.pop('model'))
        optimizer.load_state_dict(state.pop('optimizer'))
        scheduler.load_state_dict(state.pop('scheduler'))
    else:
        state = {**meta, 'epoch': 0, 'step': 0, 'loss_log': [], 'train_loss': [],
                 'heldout': [], 'jobs': [], 'timing': {}, 'complete': False}
        evaluate_heldout(0)
    gstep = state['epoch'] * n_batches + state['step']
    if len(state['loss_log']) != gstep:
        raise SystemExit(f"{state_path}: loss log does not match the step count")
    if resumed:
        print(f"Resuming from {state_path}: epoch {state['epoch'] + 1}, step {state['step']}"
              f"/{n_batches} (global {gstep}/{total})", flush=True)
    job['start_step'] = job['end_step'] = gstep
    state['jobs'].append(job)
    saved = gstep if resumed else None      # global step of the state on disk
    special_ids = torch.tensor(data['special_ids'])
    model.train()

    for epoch in range(state['epoch'], args.epochs):
        batches = length_grouped_batches(lengths, args.batch_size, args.seed, epoch,
                                         args.chunk_batches)
        padded = [len(b) * max(lengths[i] for i in b) for b in batches]
        epoch_padded = sum(padded)
        for step in range(state['step'], n_batches):
            sec_per_token = (job['step_seconds'] / job['padded_tokens']
                             if job['padded_tokens'] else 0.0)
            reason = stop_reason(args, gstep, step, padded, job['steps'], sec_per_token,
                                 state['timing'], time.perf_counter() - t0)
            if reason:
                if saved != gstep:
                    checkpoint(gstep)
                print(f"{reason} Stopped before epoch {epoch + 1} step {step + 1}/{n_batches} "
                      f"(global {gstep}/{total}); state in {state_path}. Re-run the same "
                      f"command to resume.", flush=True)
                job_summary(job, step_times, step_padded, epoch_padded, n_batches, args.epochs)
                return state, False
            t = time.perf_counter()
            ids, attention = collate([data['train'][i] for i in batches[step]], data['pad_id'])
            gen = torch.Generator().manual_seed(derive_seed(args.seed, 'mask', epoch, step))
            inputs, labels = mask_tokens(ids, special_mask(ids, attention, special_ids), gen,
                                         data['mask_id'], data['vocab_size'], args.mlm_prob)
            torch.manual_seed(derive_seed(args.seed, 'dropout', epoch, step))
            loss = mlm_loss(model, inputs, attention, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at global step {gstep}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            sec = time.perf_counter() - t
            n_tokens = int(attention.sum())
            state['loss_log'].append(loss.item())
            state['step'] = step + 1
            gstep += 1
            step_times.append(sec)
            step_padded.append(ids.numel())
            window.append((loss.item(), sec, n_tokens, ids.numel()))
            job['steps'] += 1
            job['step_seconds'] += sec
            job['tokens'] += n_tokens
            job['padded_tokens'] += ids.numel()
            if gstep % args.save_every == 0 and step + 1 < n_batches:
                checkpoint(gstep)
                saved = gstep
            if args.log_every and (gstep % args.log_every == 0 or step + 1 == n_batches):
                secs = sum(w[1] for w in window)
                mean_loss = statistics.mean(w[0] for w in window)
                print(f"  epoch {epoch + 1}/{args.epochs} step {step + 1}/{n_batches} "
                      f"(global {gstep}/{total}): loss {mean_loss:.4f}, "
                      f"lr {scheduler.get_last_lr()[0]:.2e}, {len(window) / secs:.3f} steps/s "
                      f"({secs / len(window):.2f} s/step), "
                      f"{sum(w[2] for w in window) / len(window):.0f} tokens/step "
                      f"({sum(w[3] for w in window) / len(window):.0f} padded), "
                      f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
                window.clear()
        # epoch end: train loss, held-out evaluation, checkpoint
        state['train_loss'].append(statistics.mean(state['loss_log'][epoch * n_batches:gstep]))
        print(f"Epoch {epoch + 1}/{args.epochs}: mean train loss "
              f"{state['train_loss'][-1]:.4f}", flush=True)
        evaluate_heldout(epoch + 1)
        state['epoch'], state['step'] = epoch + 1, 0
        checkpoint(gstep)
        saved = gstep
    job_summary(job, step_times, step_padded, epoch_padded, n_batches, args.epochs)
    return state, True


# ---- report and command line ----

def display_path(path):
    """Path relative to the project root when inside it (no machine-specific prefix)."""
    path = os.path.abspath(path)
    rel = os.path.relpath(path, PROJECT_ROOT)
    return path if rel.startswith('..') else rel


def file_sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def prepare_corpus(args):
    """Read, filter and split the corpus; print the counts and dropped examples.

    Returns (train, heldout, info) with `info` for the report.
    """
    sentences, n_blank = read_sentences(args.corpus)
    if args.max_sentences:
        sentences = sentences[:args.max_sentences]
    kept, dropped = filter_corpus(sentences, materiominer_sentences(), args.ngram)
    train_text, heldout_text = split_heldout(kept, args.heldout_frac, args.seed)
    info = {'path': display_path(args.corpus), 'sha256': file_sha256(args.corpus),
            'normalisation': NORMALISATION, 'ngram': args.ngram,
            'materiominer_sentences': dropped['reference_sentences'],
            'materiominer_ngrams': dropped['reference_ngrams'],
            'raw': len(sentences), 'blank_lines': n_blank,
            'duplicates_dropped': len(dropped['duplicate']),
            'overlap_dropped': len(dropped['overlap']),
            'kept': len(kept), 'train': len(train_text), 'heldout': len(heldout_text),
            'dropped_duplicates': dropped['duplicate'][:N_EXAMPLES],
            'dropped_overlap': dropped['overlap'][:N_EXAMPLES]}
    print(f"Corpus {info['path']}: {len(sentences):,} sentences ({n_blank} blank lines skipped)\n"
          f"  dropped {info['duplicates_dropped']} duplicates (after normalisation) and "
          f"{info['overlap_dropped']} sharing a word {args.ngram}-gram with MaterioMiner "
          f"({info['materiominer_sentences']} sentences, {info['materiominer_ngrams']:,} "
          f"distinct {args.ngram}-grams)\n"
          f"  kept {len(kept):,}: train {len(train_text):,}, held-out {len(heldout_text):,}")

    def short(text):
        return text if len(text) <= 150 else text[:147] + '...'
    for d in dropped['overlap'][:5]:
        print(f"  overlap [{d['index']}] '{d['ngram']}' ({', '.join(d['docs'])}): "
              f"{short(d['sentence'])}")
    for d in dropped['duplicate'][:3]:
        print(f"  duplicate [{d['index']}] of [{d['duplicate_of']}]: {short(d['sentence'])}")
    return train_text, heldout_text, info


def tokenize_sentences(tokenizer, sentences, max_length):
    """Token ids with [CLS] and [SEP], truncated to max_length, and the number truncated."""
    if not sentences:
        return [], 0
    ids = tokenizer(sentences, truncation=True, max_length=max_length)['input_ids']
    at_limit = [sentences[i] for i, x in enumerate(ids) if len(x) == max_length]
    full = tokenizer(at_limit)['input_ids'] if at_limit else []
    return ids, sum(len(x) > max_length for x in full)


def write_report(path, state, corpus, args):
    """Write the report JSON (see the module docstring)."""
    jobs = state['jobs']
    report = {
        'model': state['settings']['model'],
        'output_dir': display_path(args.out),
        'corpus': corpus,
        'settings': state['settings'],
        'settings_hash': state['settings_hash'],
        'steps': state['steps'],
        'mlm_head_loading': state['load_info'],
        'masking': state['masking'],
        'heldout': [{'epoch': h['epoch'], 'loss': round(h['loss'], 4),
                     'perplexity': round(h['perplexity'], 3),
                     'masked_tokens': h['masked_tokens'], 'seconds': h['seconds']}
                    for h in state['heldout']],
        'train_loss_per_epoch': [round(x, 4) for x in state['train_loss']],
        'train_loss_steps': [round(x, 4) for x in state['loss_log']],
        'runtime': {
            'wall_seconds': round(sum(j['wall_seconds'] for j in jobs), 1),
            'step_seconds': round(sum(j['step_seconds'] for j in jobs), 1),
            'eval_seconds': round(sum(j['eval_seconds'] for j in jobs), 1),
            'save_seconds': round(sum(j['save_seconds'] for j in jobs), 1),
            'jobs': [{k: round(v, 2) if isinstance(v, float) else v for k, v in j.items()}
                     for j in jobs],
        },
        'versions': library_versions(),
        'notes': ('heldout epoch 0 is before training; train_loss_per_epoch is the mean of the '
                  'step losses (mean cross-entropy over the masked tokens of a batch); '
                  'wall_seconds sums the jobs up to their last checkpoint'),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    return report


def run(args):
    t0 = time.perf_counter()
    import torch
    import transformers
    from transformers import AutoModelForMaskedLM, AutoTokenizer, set_seed

    transformers.utils.logging.disable_progress_bar()
    torch.set_num_threads(args.threads or os.cpu_count())
    set_seed(args.seed)
    print(f"=== DAPT of {args.model}: {torch.get_num_threads()} threads, out {args.out} ===",
          flush=True)

    train_text, heldout_text, corpus = prepare_corpus(args)
    if not train_text:
        raise SystemExit('no training sentences left')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ids, train_cut = tokenize_sentences(tokenizer, train_text, args.max_length)
    heldout_ids, heldout_cut = tokenize_sentences(tokenizer, heldout_text, args.max_length)
    corpus.update(train_tokens=sum(map(len, train_ids)), heldout_tokens=sum(map(len, heldout_ids)),
                  truncated_train=train_cut, truncated_heldout=heldout_cut)
    print(f"  {corpus['train_tokens']:,} train tokens, {corpus['heldout_tokens']:,} held-out "
          f"tokens; {train_cut} + {heldout_cut} sentences truncated to {args.max_length}",
          flush=True)

    settings = {k: getattr(args, k) for k in TRAJECTORY_ARGS}
    settings.update(state_format=STATE_FORMAT, data_sha256=hashlib.sha256(
        json.dumps([train_text, heldout_text]).encode()).hexdigest())
    digest = settings_hash(settings)
    n_batches = math.ceil(len(train_ids) / args.batch_size)
    steps = {'per_epoch': n_batches, 'total': args.epochs * n_batches,
             'warmup': math.ceil(args.warmup_ratio * args.epochs * n_batches)}

    state_path = os.path.join(args.out, 'state', STATE_FILE)
    state = load_state(state_path)
    if state is not None:
        check_resumable(state, settings, digest, state_path)
    elif os.path.exists(os.path.join(args.out, 'config.json')):
        raise SystemExit(f"{args.out} holds a model but no training state; use another --out")
    if state is not None and state.get('complete'):
        write_report(args.report, state, corpus, args)
        print(f"Training already complete ({state_path}); rewrote {args.report}")
        return

    model, info = AutoModelForMaskedLM.from_pretrained(args.model, output_loading_info=True)
    load_info = {k: sorted(map(str, v)) for k, v in info.items()}
    if load_info.get('missing_keys') or load_info.get('mismatched_keys'):
        raise SystemExit(f"the MLM head did not load cleanly: {load_info}")
    print(f"Loaded {type(model).__name__} from {args.model}: no missing or mismatched keys "
          f"({len(load_info.get('unexpected_keys', []))} unexpected)", flush=True)
    vocab_size = len(tokenizer)
    if vocab_size > model.config.vocab_size:
        raise SystemExit(f"tokenizer ({vocab_size}) larger than the model vocabulary")
    data = {'train': train_ids, 'special_ids': list(tokenizer.all_special_ids),
            'pad_id': tokenizer.pad_token_id, 'mask_id': tokenizer.mask_token_id,
            'vocab_size': vocab_size,
            'heldout': heldout_examples(heldout_ids, tokenizer.all_special_ids,
                                        tokenizer.pad_token_id, tokenizer.mask_token_id,
                                        vocab_size, args.seed, args.mlm_prob)}
    meta = {'settings': settings, 'settings_hash': digest, 'steps': steps, 'load_info': load_info,
            'masking': {'never_masked': list(tokenizer.all_special_tokens),
                        'random_ids_from': vocab_size}}
    state, done = train(model, data, args, meta, state, state_path, t0)
    if not done:
        return

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    state['complete'] = True
    state['jobs'][-1]['wall_seconds'] = round(time.perf_counter() - t0, 1)
    save_state(state_path, state)       # the weights are in --out now; the state keeps the logs
    write_report(args.report, state, corpus, args)
    print(f"Saved the model and tokenizer to {args.out} and the report to {args.report}",
          flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Domain-adaptive MLM pre-training of MatSciBERT on the PubMed sentences')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--corpus', default=DEFAULT_CORPUS)
    parser.add_argument('--out', default=DEFAULT_OUT,
                        help='final model and tokenizer; the training state goes to <out>/state')
    parser.add_argument('--report', default=DEFAULT_REPORT)
    fixed = parser.add_argument_group('pre-registered settings (do not tune)')
    fixed.add_argument('--seed', type=int, default=0)
    fixed.add_argument('--epochs', type=int, default=3)
    fixed.add_argument('--batch-size', type=int, default=32)
    fixed.add_argument('--lr', type=float, default=5e-5)
    fixed.add_argument('--weight-decay', type=float, default=0.01)
    fixed.add_argument('--warmup-ratio', type=float, default=0.06)
    fixed.add_argument('--max-grad-norm', type=float, default=1.0)
    fixed.add_argument('--max-length', type=int, default=128)
    fixed.add_argument('--mlm-prob', type=float, default=0.15)
    fixed.add_argument('--heldout-frac', type=float, default=0.05)
    fixed.add_argument('--ngram', type=int, default=8,
                       help='leakage filter: drop sentences sharing an n-gram with MaterioMiner')
    fixed.add_argument('--chunk-batches', type=int, default=50,
                       help='length grouping: sort within chunks of this many batches')
    running = parser.add_argument_group('running, smoke tests and benchmarks')
    running.add_argument('--threads', type=int, default=0, help='torch threads (0: all cores)')
    running.add_argument('--save-every', type=int, default=100)
    running.add_argument('--time-budget-min', type=float, default=0,
                         help='checkpoint and exit once the next checkpoint would end later')
    running.add_argument('--log-every', type=int, default=25)
    running.add_argument('--max-steps', type=int, default=0,
                         help='stop after this many steps in total (the schedule is unchanged)')
    running.add_argument('--max-sentences', type=int, default=0,
                         help='use only the first N corpus sentences (smoke tests)')
    args = parser.parse_args(argv)
    if args.save_every < 1:
        parser.error('--save-every must be at least 1')
    return args


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
