"""
span_ner.py - a span head for NER, the SOTA plan's P1 candidate E (docs/ROADMAP.md §4)

Every span of a sentence up to a width cap is classified on its own, with a sigmoid per entity
type, so nested and overlapping entities need no tag scheme:
  - a span's features are the first sub-word vectors of its first and last words and a learned
    embedding of its width (WIDTH_DIM);
  - then a hidden layer (HIDDEN) with GELU and dropout (DROPOUT), and one logit per type;
  - the loss is the binary cross-entropy per span and type, averaged over every (span, type)
    pair of the batch's spans up to the cap: there is no negative sampling;
  - the cap is the smallest width that covers COVERAGE of the training part's entities
    (`width_cap`), measured on that part only;
  - decoding keeps every span and type above THRESHOLD (`decode`, nested: MuLMS), or the
    non-overlapping ones, greedily from the most probable (flat: SOFC-Exp).

A sentence is {'tokens': words, 'entities': {(type, first word, last word)}}, the words
counted from 0 and the last one included. A word that the tokenizer erases has no vector: a
span starts and ends on words that have one, and its width counts every word it covers.
`ner_baseline.py --span-head` (SOFC-Exp) and `mulms_nested.py --span-head` (MuLMS) train it,
each with its own recipe; this module holds the head, the encoding, the batches, the loss and
the decoding.
"""

import math

WIDTH_DIM = 32
HIDDEN = 256
DROPOUT = 0.1
HEAD_LR = 1e-3
THRESHOLD = 0.5
COVERAGE = 0.995


def width_cap(sentences, coverage=COVERAGE):
    """The smallest width (in words) covering `coverage` of the sentences' entities."""
    widths = sorted(last - first + 1 for s in sentences for _, first, last in s['entities'])
    if not widths:
        return 1
    return widths[max(0, math.ceil(coverage * len(widths)) - 1)]


def iob2_entities(tags):
    """{(type, first, last)} of IOB2 tags, read as seqeval's default mode reads them."""
    import ner_baseline as nb
    return {(kind, start, end - 1) for start, end, kind in nb.entity_spans(tags)}


def entities_to_iob2(entities, n):
    """IOB2 tags of n words from non-overlapping entities."""
    tags = ['O'] * n
    for kind, first, last in sorted(entities, key=lambda e: (e[1], e[2])):
        if any(t != 'O' for t in tags[first:last + 1]):
            raise ValueError(f'overlapping entity {(kind, first, last)}')
        tags[first] = f'B-{kind}'
        for i in range(first + 1, last + 1):
            tags[i] = f'I-{kind}'
    return tags


def encode(sentence, tokenizer, type_index, cap, max_length=512, left=(), right=()):
    """The features of a sentence: input_ids, `first` (each kept word's first sub-word
    position), `kept` (the kept words), `spans` (pairs of positions in `kept`), `widths`, the
    targets (`labels`: per span, the type ids of the gold entities with its boundaries), and
    the counts of erased and truncated words and of gold entities that no span reaches. left
    and right: context words around the sentence (document context), never scored."""
    tokens = list(sentence['tokens'])
    words = list(left) + tokens + list(right)
    first_word, end_word = len(left), len(left) + len(tokens)
    enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length)
    first = {}
    for pos, w in enumerate(enc.word_ids()):
        if w is not None and first_word <= w < end_word and w not in first:
            first[w - first_word] = pos
    full = tokenizer(tokens, is_split_into_words=True, truncation=False)
    present = {w for w in full.word_ids() if w is not None}
    kept = sorted(first)
    spans, widths = [], []
    for a in range(len(kept)):
        for b in range(a, len(kept)):
            width = kept[b] - kept[a] + 1
            if width > cap:
                break
            spans.append((a, b))
            widths.append(width - 1)
    at = {(kept[a], kept[b]): k for k, (a, b) in enumerate(spans)}
    labels = [[] for _ in spans]
    unreachable = 0
    for kind, i, j in sentence.get('entities', ()):
        k = at.get((i, j))
        if k is None:
            unreachable += 1
        else:
            labels[k].append(type_index[kind])
    return {'input_ids': enc['input_ids'], 'first': [first[w] for w in kept], 'kept': kept,
            'spans': spans, 'widths': widths, 'labels': labels,
            'erased': len(tokens) - len(present), 'truncated': len(present) - len(kept),
            'unreachable': unreachable}


def collate(items, pad_id, n_types, device):
    """Padded tensors of a batch: input_ids, attention_mask, first (-1 where no word), spans,
    widths, span_mask and the multi-hot targets (B, S, T)."""
    import torch
    width = max(len(f['input_ids']) for f in items)
    words = max(max(len(f['first']) for f in items), 1)
    n_spans = max(max(len(f['spans']) for f in items), 1)
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    att = torch.zeros((len(items), width), dtype=torch.long)
    first = torch.full((len(items), words), -1, dtype=torch.long)
    spans = torch.zeros((len(items), n_spans, 2), dtype=torch.long)
    widths = torch.zeros((len(items), n_spans), dtype=torch.long)
    mask = torch.zeros((len(items), n_spans), dtype=torch.bool)
    targets = torch.zeros((len(items), n_spans, n_types))
    for b, f in enumerate(items):
        ids[b, :len(f['input_ids'])] = torch.tensor(f['input_ids'])
        att[b, :len(f['input_ids'])] = 1
        if f['first']:
            first[b, :len(f['first'])] = torch.tensor(f['first'])
        if f['spans']:
            spans[b, :len(f['spans'])] = torch.tensor(f['spans'])
            widths[b, :len(f['widths'])] = torch.tensor(f['widths'])
            mask[b, :len(f['spans'])] = True
        for k, types in enumerate(f['labels']):
            for t in types:
                targets[b, k, t] = 1.0
    return {'input_ids': ids.to(device), 'attention_mask': att.to(device),
            'first': first.to(device), 'spans': spans.to(device), 'widths': widths.to(device),
            'span_mask': mask.to(device), 'targets': targets.to(device)}


_HEAD = None


def make_tagger(encoder, n_types, cap):
    """The encoder with a span head (SpanTagger)."""
    global _HEAD
    if _HEAD is None:
        _HEAD = _define()
    return _HEAD(encoder, n_types, cap)


def _define():
    import torch

    class SpanTagger(torch.nn.Module):
        """Encoder, then per span [first word; last word; width embedding] -> hidden layer
        (GELU, dropout) -> one logit per type. `head` holds the parameters outside the
        encoder (trained at HEAD_LR)."""

        def __init__(self, encoder, n_types, cap):
            super().__init__()
            self.encoder = encoder
            size = encoder.config.hidden_size
            self.head = torch.nn.ModuleDict({
                'width': torch.nn.Embedding(cap, WIDTH_DIM),
                'mlp': torch.nn.Sequential(torch.nn.Linear(2 * size + WIDTH_DIM, HIDDEN),
                                           torch.nn.GELU(), torch.nn.Dropout(DROPOUT),
                                           torch.nn.Linear(HIDDEN, n_types))})

        def forward(self, batch):
            hidden = self.encoder(input_ids=batch['input_ids'],
                                  attention_mask=batch['attention_mask']).last_hidden_state
            size = hidden.size(2)
            index = batch['first'].clamp(min=0).unsqueeze(2).expand(-1, -1, size)
            words = hidden.gather(1, index)                                    # (B, W, H)
            spans = batch['spans']

            def at(k):
                return words.gather(1, spans[..., k].unsqueeze(2).expand(-1, -1, size))
            features = torch.cat([at(0), at(1), self.head['width'](batch['widths'])], dim=2)
            return self.head['mlp'](features)                                  # (B, S, T)

    return SpanTagger


def loss(logits, batch):
    """The binary cross-entropy averaged over every (span, type) pair of the batch's spans."""
    import torch
    mask = batch['span_mask'].unsqueeze(2).expand_as(logits)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits[mask], batch['targets'][mask])


def decode(logits, items, types, nested, threshold=THRESHOLD):
    """Per sentence, its predicted entities {(type, first, last)}: every span and type above
    `threshold` (nested), or the non-overlapping ones, greedily from the most probable."""
    import torch
    probs = torch.sigmoid(logits).detach().cpu()
    out = []
    for b, f in enumerate(items):
        found = []
        n = len(f['spans'])
        if n:
            hits = (probs[b, :n] > threshold).nonzero().tolist()
            for k, t in hits:
                a, c = f['spans'][k]
                found.append((float(probs[b, k, t]), types[t], f['kept'][a], f['kept'][c]))
        if nested:
            out.append({(kind, i, j) for _, kind, i, j in found})
            continue
        taken, used = set(), set()
        for p, kind, i, j in sorted(found, key=lambda x: (-x[0], x[2], x[3], x[1])):
            if used.isdisjoint(range(i, j + 1)):
                taken.add((kind, i, j))
                used.update(range(i, j + 1))
        out.append(taken)
    return out


def predict(model, feats, types, nested, batch_size, pad_id, device):
    """The decoded entities of every sentence of `feats`."""
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(feats), batch_size):
            items = feats[start:start + batch_size]
            if not any(f['spans'] for f in items):
                out += [set() for _ in items]
                continue
            batch = collate(items, pad_id, len(types), device)
            out += decode(model(batch), items, types, nested)
    return out
