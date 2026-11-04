"""
gliner_baseline.py - a free generalist NER baseline, GLiNER, on the CRF head's held-out units
(ROADMAP §4, the week-21 protocol)

GLiNER (Zaratiana et al., NAACL 2024) scores word spans against entity types given as words.
This script runs urchade/gliner_medium-v2.1 (revision MODEL_REVISION; its backbone
microsoft/deberta-v3-base at BACKBONE_REVISION; gliner GLINER_VERSION) on the four held-out
settings of the CRF head, with the CRF head's own splits:

  --granularity fine|coarse --leave-one-paper-out            MaterioMiner's 4 papers
  --granularity mulms --leave-one-subfield-out --train-pool N MuLMS's 7 subfields
  --granularity sofc --document-folds K                       SOFC-Exp's article folds

  - Splits: ner_baseline.make_folds with the CRF head's arguments for the protocol
    (crf_arguments), so every unit's training, validation and test positions are the CRF head's.
  - Labels (LABELS, fixed by the protocol): each code has one label, one to one, given in the
    corpus's label order; every example carries the full label set, the CRF head's closed set.
  - Tokens: the corpus words are GLiNER's tokens, through the package's token-level path
    (prepare_base_input, collate_batch, decode_batch); no text is split again. Decoding is flat
    (FLAT_NER, MULTI_LABEL), max_width MAX_WIDTH, max_len MAX_LEN.
  - Scoring: the spans become IOB2 tags on the same words (spans_to_iob2) and are scored with
    ner_baseline.score, seqeval as for every system.
  - Arms. --zero-shot (variant gliner_zs): one deterministic run per unit, no training; one
    decode at 0.3 is filtered by score at each threshold of ZS_THRESHOLDS, and the threshold of
    the best validation micro F1 (ties: the higher threshold) decodes the test sentences.
    Fine-tuning (variant gliner_ft): from the pretrained checkpoint, on the unit's training
    sentences, in float32, under ner_baseline.cuda_determinism on a CUDA device: AdamW with the
    encoder at LR_ENCODER and the other parameters at LR_OTHERS, weight decay WEIGHT_DECAY, a
    linear schedule with warm-up over WARMUP_RATIO of the steps, batches of BATCH_SIZE (or that
    many sentences in --grad-accumulation micro-batches), at most --epochs (EPOCHS) epochs with
    patience PATIENCE, the snapshot of the epoch of best validation micro F1 at FT_THRESHOLD,
    which is also the test threshold; the model seed is the CRF head's (seeds[0] + --seed-offset).
    The training loop is this module's own (fine_tune): the package's Trainer needs accelerate,
    which the pinned environment lacks. What the protocol leaves open follows the package's
    training defaults (GLiNER.create_training_args and its Trainer): the loss (LOSS: focal with
    alpha -1 and gamma 0, i.e. binary cross-entropy, summed), gradient clipping at MAX_GRAD_NORM,
    no weight decay on biases and LayerNorm weights, AdamW's betas and epsilon, and warm-up steps
    rounded up. The training sentences are shuffled per epoch as the CRF head's are
    (random.Random(seed * 1000 + epoch)).
  - Pins: the model snapshot and the backbone (whose tokenizer and configuration GLiNER loads by
    name) must be the pinned revisions in the Hugging Face cache (kaggle_jobs.py pins both,
    run_jobs.pin_models); every weight comes from the checkpoint (strict loading).
  - Two exploratory checks of week 21, for a referee's two points (outside the decision count;
    their analysis is ner_seed_analysis.py's week21x, fixed before their runs):
    E1, --coarse-names (variant gliner_ftn): gliner_ft's fine-tuning on MaterioMiner's coarse
    level with one change, the labels: each code is given as its class name in words
    (COARSE_NAMES, with the source of every name) instead of the code lowercased.
    E2, --package-trainer (variant gliner_ftp): the fine-tuning by the package's own trainer,
    gliner.training's Trainer and TrainingArguments, in place of this module's loop, with
    gliner_ft's recipe wherever the package's arguments allow it (PACKAGE_ARGS), for a fixed
    number of epochs per unit: the seed's best epoch in the stored gliner_ft run of the same unit
    and seed offset (stored_epochs, from --stored-runs, results/gpu by default). The linear
    schedule keeps gliner_ft's steps (num_train_epochs EPOCHS; a callback stops the training after
    the fixed epochs, stop_after), and the test sentences are decoded and scored at FT_THRESHOLD
    as gliner_ft's are. The package's trainer needs accelerate, at ACCELERATE_VERSION (checked,
    and recorded in the config with the trainer).

Results go to <PROCESSED_DIR>/ner/ner_results_<granularity>_<protocol>_<variant>.json, the
variant gliner_ft, gliner_zs, gliner_ftn or gliner_ftp with the tag (gliner_ft_m300), in the CRF
head's schema (labels, config with the protocol keys and the pins, per run f1, precision, recall,
f1_strict, per_class, seed, best_epoch, val_f1, n_train, n_val, n_test, history, test_index,
test_gold, test_pred, run_id, runtime_sec and, on a GPU, environment; a zero-shot run adds its
threshold, an E2 run the epochs it trained). The file is written after every unit and finished
units are skipped when re-run, as ner_baseline.py does; a file written with other settings is not
resumed (--overwrite replaces it; smoke-test files, --max-train, are replaced at will).

Usage:
  python src/graph/gliner_baseline.py --granularity fine --leave-one-paper-out --tag m0
  python src/graph/gliner_baseline.py --granularity mulms --leave-one-subfield-out \\
      --train-pool 400 --tag m300 --seed-offset 300
  python src/graph/gliner_baseline.py --granularity sofc --document-folds 5 --zero-shot --tag m0
  python src/graph/gliner_baseline.py --granularity coarse --leave-one-paper-out --coarse-names \\
      --tag m300 --seed-offset 300                                                      # E1
  python src/graph/gliner_baseline.py --granularity fine --leave-one-paper-out --package-trainer \\
      --units 10.1007_s11340-021-00758-x --tag m300 --seed-offset 300                   # E2
"""

import argparse
import json
import math
import os
import random
import re
import struct
import sys
import time
from pathlib import Path

import ner_baseline as nb
from ner_ensemble import spans_to_tags, tags_to_spans

GLINER_VERSION = '0.2.30'
MODEL = 'urchade/gliner_medium-v2.1'
MODEL_REVISION = '40ec419335d09393f298636f471328b722c6da9e'
BACKBONE = 'microsoft/deberta-v3-base'
BACKBONE_REVISION = '8ccc9b6f36199bec6961081d44eb72fb3f7353f3'
# decoding (the protocol)
MAX_WIDTH = 12
MAX_LEN = 384
FLAT_NER = True
MULTI_LABEL = False
ZS_THRESHOLDS = (0.3, 0.4, 0.5, 0.6)    # the zero-shot grid; one decode at the lowest
FT_THRESHOLD = 0.5                       # fine-tuning: the epoch's and the test threshold
EVAL_BATCH_SIZE = 8                      # the package's inference batch size
# fine-tuning (the protocol)
LR_ENCODER = 1e-5
LR_OTHERS = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
BATCH_SIZE = 8
EPOCHS = 20
PATIENCE = 3
# fine-tuning, what the protocol leaves open: the package's training defaults
# (GLiNER.create_training_args, its Trainer and TrainingArguments)
MAX_GRAD_NORM = 1.0
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
LOSS = {'alpha': -1.0, 'gamma': 0.0, 'rel_alpha': -1.0, 'rel_gamma': 0.0, 'prob_margin': 0.0,
        'label_smoothing': 0.0, 'reduction': 'sum', 'negatives': 1.0, 'masking': 'none',
        'loss_type': 'focal', 'use_span_width_weight': False, 'dice_gamma': 1.0,
        'ce_calib_weight': 0.5}
ENCODER = 'token_rep_layer'              # the parameters at LR_ENCODER (the package's rule)
VARIANTS = {False: 'gliner_ft', True: 'gliner_zs'}

# The labels (the protocol): MaterioMiner's fine-grained classes camel case split and lowercased,
# with the dataset's misspelled CrackInitation spelled out; its coarse-grained codes lowercased
# (no documented expansion, so no zero-shot arm); SOFC-Exp's types; MuLMS's from its paper.
FINE_SPELLING = {'CrackInitation': 'crack initiation'}
SOFC_LABELS = {'DEVICE': 'device', 'EXPERIMENT': 'experiment', 'MATERIAL': 'material',
               'VALUE': 'value'}
MULMS_LABELS = {'CITE': 'citation', 'DEV': 'device', 'FORM': 'form', 'INSTRUMENT': 'instrument',
                'MAT': 'material', 'MEASUREMENT': 'measurement', 'NUM': 'number',
                'PROPERTY': 'property', 'RANGE': 'range', 'SAMPLE': 'sample',
                'TECHNIQUE': 'technique', 'UNIT': 'unit', 'VALUE': 'value'}
ZERO_SHOT_GRANULARITIES = ('fine', 'mulms', 'sofc')

# The exploratory checks of week 21 (module docstring): E1 and E2's variants
NAMES_VARIANT = 'gliner_ftn'
PACKAGE_VARIANT = 'gliner_ftp'
# E1: MaterioMiner's 27 coarse-grained codes as the names of the classes they stand for, camel case
# split and lowercased. The source of every name is the MaterioMiner paper (Durmaz et al. 2024,
# Scientific Data 11, 1112, doi:10.1038/s41597-024-03926-5):
#   - its Fig. 7b names the 27 coarse-grained entity types, ontology classes in camel case (the
#     class each code stands for is in the comment beside it);
#   - its Methods ("Data post-processing") give two codes in words, "mm:Mechanism is abbreviated
#     as MEC, mm:EnvironmentalTestParameter as ETP";
#   - its Data Records give "a mapping of the tags to the ontological preferred labels" in the
#     release's CG-NER_tags.csv (v1.0.1, doi:10.24406/fordatis/329.2,
#     dataset/coarse_grained_ner/), which has every code below; that release's CoNLL files are
#     byte-identical to data/raw/dataset's.
# Checked against the dataset files: each code's entities in data/raw/dataset/coarse_grained_ner
# number its bar in Fig. 7b (2,194 in all; CPR's 7 has no printed number), and each name is the
# rdfs:label, lowercased, of the class it names in data/raw/dataset/ontologies/ontology.ttl. So
# no name rests on the ontology alone, and none on the fallback (the most frequent fine-grained
# class nested in the code's spans). (The week-21 protocol took the codes to have no documented
# expansion; the paper and its release document one.)
COARSE_NAMES = {
    'CEL': 'chemical element',              # ChemicalElement
    'CGE': 'crystallographic entity',       # CrystallographicEntity
    'CHP': 'characterization process',      # CharacterizationProcess
    'CPR': 'chemical property',             # ChemicalProperty
    'DAM': 'damage',                        # Damage
    'DAT': 'data set',                      # DataSet
    'DEF': 'defect',                        # Defect
    'EQU': 'equipment',                     # Equipment
    'ETP': 'environmental test parameter',  # EnvironmentalTestParameter; also the paper's text
    'LOC': 'location',                      # Location
    'MAC': 'material class',                # MaterialClass
    'MAT': 'material',                      # Material
    'MEC': 'mechanism',                     # Mechanism; also the paper's text
    'MEP': 'mechanical property',           # MechanicalProperty
    'MFP': 'manufacturing process',         # ManufacturingProcess
    'MIC': 'microstructure',                # Microstructure
    'MIP': 'microstructural property',      # MicrostructuralProperty
    'MOD': 'model',                         # Model
    'MTP': 'mechanical test parameter',     # MechanicalTestParameter
    'OBJ': 'object',                        # Object
    'ODE': 'orientation descriptor',        # OrientationDescriptor
    'PQT': 'physical quantity',             # PhysicalQuantity
    'PRO': 'property',                      # Property
    'PRP': 'process parameter',             # ProcessParameter
    'SDE': 'shape descriptor',              # ShapeDescriptor
    'STD': 'state descriptor',              # StateDescriptor
    'VAL': 'value',                         # Value
}
# E2: the package's trainer (gliner.training's Trainer and TrainingArguments) needs accelerate,
# installed --no-deps beside the pinned packages at this version (kaggle_jobs.py, W21X_PINS); a run
# checks it and records it in the config with the trainer
PACKAGE_TRAINER = 'gliner.training.Trainer'
ACCELERATE_VERSION = '1.15.0'
# E2's epochs: per unit, the seed's best epoch in the stored gliner_ft run (stored_epochs)
STORED_RUNS = Path(__file__).resolve().parents[2] / 'results' / 'gpu'
# E2's TrainingArguments (besides the output directory, the seed and the device): gliner_ft's recipe
# wherever the package's arguments allow it. The encoder (token_rep_layer, the package's rule) at
# learning_rate, the rest at others_lr, weight decay on both (never on biases and LayerNorm weights,
# the package's rule); AdamW's betas and epsilon; the linear schedule over num_train_epochs EPOCHS,
# so that its steps and its warm-up (WARMUP_RATIO of them, rounded up) are gliner_ft's, a run
# stopping after its fixed epochs (stop_after); batches of BATCH_SIZE, no accumulation; clipping at
# MAX_GRAD_NORM; LOSS's loss (masking 'none', as GLiNER.create_training_args has it: the
# TrainingArguments' default 'global' draws random numbers at every step, though with negatives 1.0
# it masks nothing). No evaluation, checkpoint or report during the training; the batches are
# collated in the main process; the precision is float32, the determinism ner_baseline's
# (cuda_determinism, not the trainer's full_determinism, which would reset the cuBLAS workspace).
PACKAGE_ARGS = {
    'num_train_epochs': EPOCHS, 'max_steps': -1, 'per_device_train_batch_size': BATCH_SIZE,
    'gradient_accumulation_steps': 1, 'optim': 'adamw_torch', 'learning_rate': LR_ENCODER,
    'others_lr': LR_OTHERS, 'weight_decay': WEIGHT_DECAY, 'others_weight_decay': WEIGHT_DECAY,
    'adam_beta1': ADAM_BETAS[0], 'adam_beta2': ADAM_BETAS[1], 'adam_epsilon': ADAM_EPS,
    'lr_scheduler_type': 'linear', 'warmup_ratio': WARMUP_RATIO, 'warmup_steps': 0,
    'max_grad_norm': MAX_GRAD_NORM, 'focal_loss_alpha': LOSS['alpha'],
    'focal_loss_gamma': LOSS['gamma'], 'rel_focal_loss_alpha': LOSS['rel_alpha'],
    'rel_focal_loss_gamma': LOSS['rel_gamma'], 'focal_loss_prob_margin': LOSS['prob_margin'],
    'label_smoothing': LOSS['label_smoothing'], 'loss_reduction': LOSS['reduction'],
    'negatives': LOSS['negatives'], 'masking': LOSS['masking'], 'loss_type': LOSS['loss_type'],
    'use_span_width_weight': LOSS['use_span_width_weight'], 'dice_gamma': LOSS['dice_gamma'],
    'ce_calib_weight': LOSS['ce_calib_weight'], 'fp16': False, 'bf16': False,
    'full_determinism': False, 'eval_strategy': 'no', 'save_strategy': 'no',
    'logging_strategy': 'epoch', 'report_to': 'none', 'disable_tqdm': True,
    'remove_unused_columns': False, 'dataloader_num_workers': 0, 'dataloader_drop_last': False}
# the config keys of a CRF results file that fix its protocol's split (nb.stored_config)
PROTOCOL_KEYS = ('seeds', 'split_seed', 'vary_split', 'leave_one_paper_out', 'top_k',
                 'leave_one_subfield_out', 'train_pool', 'document_folds')
RESUME_IGNORED = ('device', 'threads')   # the config keys a resumed file may differ in


# ---------------------------------------------------------------------------
# Labels and the conversions between spans and tags
# ---------------------------------------------------------------------------
def camel_case_label(code):
    """A fine-grained class name camel case split and lowercased (StressAmplitude: stress
    amplitude), FINE_SPELLING's names as spelled there."""
    if code in FINE_SPELLING:
        return FINE_SPELLING[code]
    return ' '.join(w.lower() for w in re.findall(r'[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+', code))


def label_map(granularity, codes, coarse_names=False):
    """{code: label} of a corpus's codes, in their order (the protocol's labels; with
    coarse_names, E1's: MaterioMiner's coarse-grained codes as the class names of COARSE_NAMES).
    Every code needs a label and no two codes share one (ValueError otherwise)."""
    if coarse_names and granularity != 'coarse':
        raise ValueError(f'{granularity}: the class names of COARSE_NAMES are for the coarse '
                         f'level only')
    if granularity == 'fine':
        names = {c: camel_case_label(c) for c in codes}
    elif granularity == 'coarse' and not coarse_names:
        names = {c: c.lower() for c in codes}
    else:
        table = (COARSE_NAMES if coarse_names
                 else {'sofc': SOFC_LABELS, 'mulms': MULMS_LABELS}[granularity])
        missing = [c for c in codes if c not in table]
        if missing or len(codes) != len(table):
            raise ValueError(f'{granularity}: the codes {list(codes)} are not those of the '
                             f'label table {list(table)}')
        names = {c: table[c] for c in codes}
    if len(set(names.values())) != len(names) or not all(names.values()):
        raise ValueError(f'{granularity}: the labels {names} are not one to one')
    return names


def gold_spans(tags, names):
    """GLiNER's training spans of an IOB2 tag sequence: (start, end, label), both ends words
    (inclusive), the label of `names` ({code: label}); an I- tag that continues no entity of its
    type starts one, as seqeval's default mode reads it."""
    return [(start, end - 1, names[code]) for start, end, code in tags_to_spans(tags)[0]]


def spans_to_iob2(spans, n_words, codes_of):
    """IOB2 tags on a sentence's n_words words from non-overlapping (start, end, label) spans, both
    ends inclusive word positions (GLiNER's), each label mapped back to its code by codes_of
    ({label: code}); ValueError for an overlap or a span outside the sentence."""
    return spans_to_tags([(start, end + 1, codes_of[label]) for start, end, label in spans],
                         n_words)


def float32(x):
    """x rounded to float32, the precision in which the package compares scores and thresholds."""
    return struct.unpack('f', struct.pack('f', x))[0]


def above(spans, threshold):
    """The decoded spans (start, end, label, score) whose score exceeds `threshold` as the decoder
    compares it (in float32): a decode at a lower threshold filtered so is the decode at this one,
    since the flat greedy decoding takes the spans in descending order of score."""
    limit = float32(threshold)
    return [s for s in spans if s[3] > limit]


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
def pinned_snapshot(repo_id, revision, filename):
    """The local directory of a pinned snapshot, from the Hugging Face cache (no network); with
    network access and no cached copy, the file is fetched at that revision first."""
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    path = try_to_load_from_cache(repo_id, filename, revision=revision)
    if not isinstance(path, str):
        if os.environ.get('HF_HUB_OFFLINE') == '1':
            raise SystemExit(f'{repo_id} at {revision} is not in the Hugging Face cache (pin it: '
                             f'kaggle_jobs.py does, with run_jobs.pin_models)')
        path = hf_hub_download(repo_id, filename, revision=revision)
    return Path(path).parent


def check_backbone():
    """The backbone that GLiNER loads by name (its tokenizer and configuration) must resolve to
    BACKBONE_REVISION: the cache's main ref of BACKBONE points at it (run_jobs.pin_models)."""
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(BACKBONE, 'config.json')
    if not isinstance(path, str) or Path(path).parent.name != BACKBONE_REVISION:
        raise SystemExit(f'{BACKBONE}: the cached main revision is {path!r}, not the pinned '
                         f'{BACKBONE_REVISION} (pin it with run_jobs.pin_models)')


def load_model(device):
    """GLiNER at the pinned revisions, every weight from the checkpoint (strict), in float32, with
    the protocol's max_len and max_width, on `device`."""
    import gliner
    import torch
    if gliner.__version__ != GLINER_VERSION:
        raise SystemExit(f'gliner {gliner.__version__} is installed, not {GLINER_VERSION}')
    check_backbone()
    snapshot = pinned_snapshot(MODEL, MODEL_REVISION, 'gliner_config.json')
    if snapshot.name != MODEL_REVISION:
        raise SystemExit(f'{MODEL}: snapshot {snapshot}, not revision {MODEL_REVISION}')
    if not (snapshot / 'model.safetensors').is_file():
        pinned_snapshot(MODEL, MODEL_REVISION, 'model.safetensors')
    model = gliner.GLiNER.from_pretrained(str(snapshot), max_length=MAX_LEN, max_width=MAX_WIDTH,
                                          strict=True)
    config = model.config
    if (config.model_name, config.max_len, config.max_width) != (BACKBONE, MAX_LEN, MAX_WIDTH):
        raise SystemExit(f'{MODEL}: backbone {config.model_name}, max_len {config.max_len}, '
                         f'max_width {config.max_width}')
    dtypes = {p.dtype for p in model.parameters() if p.is_floating_point()}
    if dtypes != {torch.float32}:
        raise SystemExit(f'{MODEL}: parameters in {dtypes}, not float32')
    return model.to(device)


def forward_inputs(model, batch, device):
    """The entries of a collated batch that the model's forward takes, on `device` (the package's
    run_batch does the same under inference mode; here under no_grad)."""
    import torch
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            if k not in model._DECODE_ONLY_KEYS}


def decode(model, sentences, prompts, threshold, device):
    """The flat decoding of each sentence (a list of words) with the labels `prompts`: [(start,
    end, label, score)] per sentence, both ends inclusive word positions, in batches of
    EVAL_BATCH_SIZE in the sentences' order, through the package's token-level path."""
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(sentences), EVAL_BATCH_SIZE):
            chunk = [list(words) for words in sentences[start:start + EVAL_BATCH_SIZE]]
            batch = model.collate_batch(model.prepare_base_input(chunk), list(prompts))
            output = model.model(**forward_inputs(model, batch, device), threshold=threshold)
            for spans in model.decode_batch(output, batch, threshold=threshold, flat_ner=FLAT_NER,
                                            multi_label=MULTI_LABEL):
                out.append([(s.start, s.end, s.entity_type, s.score) for s in spans])
    return out


def predicted_tags(decoded, sentences, codes_of, threshold):
    """IOB2 tags per sentence of decoded spans above `threshold`."""
    return [spans_to_iob2([s[:3] for s in above(spans, threshold)], len(words), codes_of)
            for spans, words in zip(decoded, sentences, strict=True)]


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------
def zero_shot(train, val, test, labels, names, device, test_index):
    """A zero-shot run (gliner_zs): the pretrained model decodes the validation sentences once at
    the lowest threshold of ZS_THRESHOLDS; each threshold's validation micro F1 is that decode
    filtered by score; the best one (ties: the higher threshold) is the run's, at which the test
    decode (once at the lowest, filtered) is scored. history: each threshold's validation F1."""
    prompts = [names[c] for c in labels]
    codes_of = {names[c]: c for c in labels}
    model = load_model(device)
    lowest = min(ZS_THRESHOLDS)
    val_words, val_gold = [w for w, _, _ in val], [g for _, g, _ in val]
    decoded = decode(model, val_words, prompts, lowest, device)
    history = [{'threshold': t,
                'val_f1': nb.score(val_gold, predicted_tags(decoded, val_words, codes_of, t))['f1']}
               for t in ZS_THRESHOLDS]
    best = max(history, key=lambda h: (h['val_f1'], h['threshold']))
    test_words = [w for w, _, _ in test]
    test_pred = predicted_tags(decode(model, test_words, prompts, lowest, device), test_words,
                               codes_of, best['threshold'])
    result = nb.score([g for _, g, _ in test], test_pred)
    result.update({'seed': None, 'best_epoch': None, 'val_f1': best['val_f1'],
                   'threshold': best['threshold'], 'n_train': len(train), 'n_val': len(val),
                   'n_test': len(test), 'history': history})
    result.update(nb.prediction_fields(test_index, test, test_pred))
    return result


def parameter_groups(model):
    """AdamW's parameter groups, the package Trainer's with a second learning rate: the encoder's
    parameters (ENCODER in the name) at LR_ENCODER and the others at LR_OTHERS, each with
    WEIGHT_DECAY except the biases and the LayerNorm weights (no decay)."""
    import torch
    from transformers.trainer_pt_utils import get_parameter_names
    decay = {n for n in get_parameter_names(model, [torch.nn.LayerNorm]) if 'bias' not in n}
    groups = []
    for encoder, lr in ((False, LR_OTHERS), (True, LR_ENCODER)):
        for decayed in (True, False):
            params = [p for n, p in model.named_parameters()
                      if p.requires_grad and (ENCODER in n) == encoder and (n in decay) == decayed]
            if params:
                groups.append({'params': params, 'lr': lr,
                               'weight_decay': WEIGHT_DECAY if decayed else 0.0})
    return groups


def fine_tune(seed, train, val, test, labels, names, args, device, test_index):
    """A fine-tuning run (gliner_ft; module docstring): set_seed(seed), the pretrained model, then
    at most args.epochs epochs on `train`, each followed by the validation decode at FT_THRESHOLD;
    the snapshot of the best validation micro F1 (a later epoch must exceed it) decodes the test
    sentences. history: each epoch's summed training loss and validation F1."""
    import torch
    from transformers import get_linear_schedule_with_warmup, set_seed
    set_seed(seed)
    model = load_model(device)
    prompts = [names[c] for c in labels]
    codes_of = {names[c]: c for c in labels}
    examples = [{'tokenized_text': list(w), 'ner': gold_spans(t, names)} for w, t, _ in train]
    collator = model.data_collator_class(model.config, data_processor=model.data_processor,
                                         prepare_labels=True)
    optimizer = torch.optim.AdamW(parameter_groups(model), lr=LR_ENCODER, betas=ADAM_BETAS,
                                  eps=ADAM_EPS)
    steps = args.epochs * math.ceil(len(examples) / BATCH_SIZE)
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(WARMUP_RATIO * steps), steps)
    micro = BATCH_SIZE // args.grad_accumulation
    val_words, val_gold = [w for w, _, _ in val], [g for _, g, _ in val]

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}

    best = (-1.0, None, -1)
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(examples)))
        random.Random(seed * 1000 + epoch).shuffle(order)
        total = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch_ids = order[start:start + BATCH_SIZE]
            for part in range(0, len(batch_ids), micro):
                chunk = [examples[i] for i in batch_ids[part:part + micro]]
                batch = collator(chunk, entity_types=[list(prompts) for _ in chunk])
                loss = model.model(**forward_inputs(model, batch, device), **LOSS).loss
                loss.backward()          # summed over the sentences: the micro-batches add up
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        val_pred = predicted_tags(decode(model, val_words, prompts, FT_THRESHOLD, device),
                                  val_words, codes_of, FT_THRESHOLD)
        val_f1 = nb.score(val_gold, val_pred)['f1']
        history.append({'epoch': epoch, 'train_loss': round(total, 4), 'val_f1': val_f1})
        print(f"    seed {seed} epoch {epoch}: loss {total:.3f} val F1 {val_f1:.2f}", flush=True)
        if val_f1 > best[0]:
            best = (val_f1, snapshot(), epoch)
        elif epoch - best[2] >= args.patience:
            break
    model.model.load_state_dict(best[1])
    test_words = [w for w, _, _ in test]
    test_pred = predicted_tags(decode(model, test_words, prompts, FT_THRESHOLD, device),
                               test_words, codes_of, FT_THRESHOLD)
    result = nb.score([g for _, g, _ in test], test_pred)
    result.update({'seed': seed, 'best_epoch': best[2], 'val_f1': best[0], 'n_train': len(train),
                   'n_val': len(val), 'n_test': len(test), 'history': history})
    result.update(nb.prediction_fields(test_index, test, test_pred))
    return result


def check_accelerate():
    """E2: the package's trainer needs accelerate, at ACCELERATE_VERSION (SystemExit otherwise);
    returns its version."""
    try:
        import accelerate
    except ImportError as e:
        raise SystemExit(f"--package-trainer: the package's trainer needs "
                         f'accelerate=={ACCELERATE_VERSION} (kaggle_jobs.py installs it --no-deps, '
                         f'W21X_PINS): {e}') from None
    if accelerate.__version__ != ACCELERATE_VERSION:
        raise SystemExit(f'accelerate {accelerate.__version__} is installed, not '
                         f'{ACCELERATE_VERSION}')
    return accelerate.__version__


def stored_epochs(stored_dir, granularity, protocol, seed_offset, run_id, seed, names):
    """E2's number of epochs for a unit: the seed's best epoch in the stored gliner_ft run of the
    same unit and seed offset (<stored_dir>/ner_results_<granularity>_<protocol>_gliner_ft_m<offset>
    .json), that is its best_epoch + 1, since best_epoch is the 0-based index of that epoch in the
    run's history. SystemExit when the file, the unit or a run of this seed is not there, or when
    the stored run is not a full gliner_ft run with these labels."""
    variant = f'{VARIANTS[False]}_m{seed_offset}'
    path = results_path(stored_dir, granularity, protocol, variant)
    if not os.path.exists(path):
        raise SystemExit(f'--package-trainer: no stored {variant} file {path} (--stored-runs)')
    with open(path) as f:
        stored = json.load(f)
    config = stored.get('config') or {}
    if (stored.get('variant') != variant or config.get('max_train')
            or config.get('label_names') != names):
        raise SystemExit(f'--package-trainer: {path} is not a full {variant} run with these labels')
    run = next((r for r in stored['runs'] if r.get('run_id') == run_id), None)
    if run is None or run.get('seed') != seed:
        raise SystemExit(f'--package-trainer: {path} has no run of {run_id} with seed {seed}')
    best = run['best_epoch']
    if not (0 <= best < len(run['history'])) or run['history'][best]['val_f1'] != run['val_f1']:
        raise SystemExit(f'--package-trainer: {path}, {run_id}: best_epoch {best} is not the '
                         f'index of the best validation F1 in its history')
    return best + 1


def stop_after(epochs):
    """E2: a transformers TrainerCallback that stops the training at the end of its `epochs`-th
    epoch, the schedule keeping its steps (PACKAGE_ARGS)."""
    from transformers import TrainerCallback

    class StopAfter(TrainerCallback):
        def __init__(self):
            self.ended = 0

        def on_epoch_end(self, args, state, control, **kwargs):
            self.ended += 1
            if self.ended >= epochs:
                control.should_training_stop = True
            return control

    return StopAfter()


def package_fine_tune(seed, epochs, train, val, test, labels, names, device, test_index):
    """A fine-tuning run by the package's own trainer (gliner_ftp, E2; module docstring):
    set_seed(seed) and the pretrained model, as fine_tune's; then gliner.training's Trainer with
    the TrainingArguments of PACKAGE_ARGS and the seed, built as GLiNER.train_model builds it (the
    package's collator, the tokenizer as its processing class), on `train`, every example
    carrying the full label set in the corpus's order (ner_labels, the package's predefined
    labels: fine_tune's entity_types, the same class ids), for `epochs` epochs (stop_after; the
    trainer shuffles the batches by its own rule, from the seed). Then the validation and the test
    sentences are decoded at FT_THRESHOLD and scored as fine_tune's. best_epoch is the last epoch
    (epochs - 1), val_f1 its validation micro F1 (reported; it selects nothing); history: the
    trainer's log of each epoch (its mean loss per step, its last learning rate and gradient
    norm). A batch that the trainer skips (it skips one that runs out of memory, where the loop
    would stop) makes the run fail: RuntimeError."""
    import logging
    import tempfile

    from gliner.training import Trainer, TrainingArguments
    from transformers import set_seed
    set_seed(seed)
    model = load_model(device)
    prompts = [names[c] for c in labels]
    codes_of = {names[c]: c for c in labels}
    examples = [{'tokenized_text': list(w), 'ner': gold_spans(t, names),
                 'ner_labels': list(prompts)} for w, t, _ in train]
    skipped = []                # the trainer skips a batch that runs out of memory, with a warning

    class Skips(logging.Handler):
        def emit(self, record):
            if record.getMessage().startswith('Skipping batch'):
                skipped.append(record.getMessage())

    handler, trainer_log = Skips(logging.WARNING), logging.getLogger(Trainer.__module__)
    trainer_log.addHandler(handler)
    try:
        with tempfile.TemporaryDirectory() as out_dir:
            training_args = TrainingArguments(output_dir=out_dir, seed=seed,
                                              use_cpu=not str(device).startswith('cuda'),
                                              **PACKAGE_ARGS)
            trainer = Trainer(model=model, args=training_args, train_dataset=examples,
                              eval_dataset=None, data_collator=model._create_data_collator(),
                              processing_class=model.data_processor.transformer_tokenizer,
                              callbacks=[stop_after(epochs)])
            trainer.train()
    finally:
        trainer_log.removeHandler(handler)
    if skipped:                 # another recipe: no run rather than a silently different one
        raise RuntimeError(f'the trainer skipped {len(skipped)} batches: {skipped[0]}')
    logs = [e for e in trainer.state.log_history if 'loss' in e]
    if len(logs) != epochs:
        raise RuntimeError(f'the trainer logged {len(logs)} epochs, not {epochs}')
    history = [{'epoch': round(e['epoch']) - 1, 'train_loss_mean': round(float(e['loss']), 4),
                'learning_rate': float(e['learning_rate']),
                'grad_norm': (None if e.get('grad_norm') is None
                              else round(float(e['grad_norm']), 4))} for e in logs]
    val_words, val_gold = [w for w, _, _ in val], [g for _, g, _ in val]
    val_pred = predicted_tags(decode(model, val_words, prompts, FT_THRESHOLD, device), val_words,
                              codes_of, FT_THRESHOLD)
    test_words = [w for w, _, _ in test]
    test_pred = predicted_tags(decode(model, test_words, prompts, FT_THRESHOLD, device),
                               test_words, codes_of, FT_THRESHOLD)
    result = nb.score([g for _, g, _ in test], test_pred)
    result.update({'seed': seed, 'best_epoch': epochs - 1, 'epochs': epochs,
                   'val_f1': nb.score(val_gold, val_pred)['f1'], 'n_train': len(train),
                   'n_val': len(val), 'n_test': len(test), 'history': history})
    result.update(nb.prediction_fields(test_index, test, test_pred))
    return result


# ---------------------------------------------------------------------------
# Runs and results files
# ---------------------------------------------------------------------------
def crf_arguments(args):
    """The CRF head's arguments (ner_baseline.parse_args) of this protocol: make_folds builds the
    units from them exactly as for the stored CRF runs."""
    protocol = {'fine': ['--leave-one-paper-out'], 'coarse': ['--leave-one-paper-out'],
                'mulms': ['--leave-one-subfield-out', '--train-pool', str(args.train_pool)],
                'sofc': ['--document-folds', str(args.document_folds)]}[args.granularity]
    return nb.parse_args(['--granularity', args.granularity, *protocol, '--split-seed',
                          str(args.split_seed), '--crf'])


def variant_name(args):
    """gliner_ft or gliner_zs (E1: gliner_ftn, E2: gliner_ftp), then the tag (gliner_ft_m300)."""
    if args.coarse_names:
        return f'{NAMES_VARIANT}_{args.tag}'
    if args.package_trainer:
        return f'{PACKAGE_VARIANT}_{args.tag}'
    return f'{VARIANTS[args.zero_shot]}_{args.tag}'


def results_path(out_dir, granularity, protocol, variant, max_train=0):
    """<out_dir>/ner_results_<granularity>_<protocol>_<variant>[_smoke<N>].json, ner_baseline's
    naming."""
    smoke = f'_smoke{max_train}' if max_train else ''
    return os.path.join(out_dir, f'ner_results_{granularity}'
                                 f'{nb.result_suffix(protocol, variant)}{smoke}.json')


def stored_config(args, crf, names):
    """The results file's config: the CRF head's protocol keys (PROTOCOL_KEYS), the pins (gliner
    version and both revisions), the labels, the decoding, the arm's recipe and the run options.
    E1's is gliner_ft's with its labels; E2's recipe is the trainer, the accelerate version, the
    stored runs whose best epochs it trains (epochs_from) and the trainer's arguments."""
    config = {k: v for k, v in nb.stored_config(crf).items() if k in PROTOCOL_KEYS}
    config.update({'model': MODEL, 'model_revision': MODEL_REVISION, 'backbone': BACKBONE,
                   'backbone_revision': BACKBONE_REVISION, 'gliner_version': GLINER_VERSION,
                   'zero_shot': args.zero_shot, 'label_names': dict(names), 'max_width': MAX_WIDTH,
                   'max_len': MAX_LEN, 'flat_ner': FLAT_NER, 'multi_label': MULTI_LABEL,
                   'dtype': 'float32', 'eval_batch_size': EVAL_BATCH_SIZE})
    if args.zero_shot:
        config.update({'thresholds': list(ZS_THRESHOLDS), 'threshold_ties': 'higher'})
    elif args.package_trainer:
        config.update({'threshold': FT_THRESHOLD, 'trainer': PACKAGE_TRAINER,
                       'accelerate_version': ACCELERATE_VERSION,
                       'epochs_from': f'{VARIANTS[False]}_m{args.seed_offset}: best_epoch + 1',
                       'training_args': dict(PACKAGE_ARGS)})
    else:
        config.update({'threshold': FT_THRESHOLD, 'epochs': args.epochs, 'patience': args.patience,
                       'batch_size': BATCH_SIZE, 'grad_accumulation': args.grad_accumulation,
                       'lr_encoder': LR_ENCODER, 'lr_others': LR_OTHERS,
                       'weight_decay': WEIGHT_DECAY, 'warmup_ratio': WARMUP_RATIO,
                       'max_grad_norm': MAX_GRAD_NORM, 'adam_betas': list(ADAM_BETAS),
                       'adam_eps': ADAM_EPS, 'loss': dict(LOSS)})
    config.update({'tag': args.tag, 'seed_offset': args.seed_offset, 'save_predictions': True,
                   'max_train': args.max_train, 'device': args.device, 'threads': args.threads})
    return config


def check_resume(path, labels, variant, args, config):
    """ner_baseline.check_overwrite for the file, and its config must be this one but for
    RESUME_IGNORED; OverwriteError otherwise (nothing is checked with --overwrite or
    --max-train, whose files are replaced at will)."""
    if args.overwrite or args.max_train or not os.path.exists(path):
        return
    nb.check_overwrite(path, labels, MODEL, variant, args.seed_offset, save_predictions=True)
    with open(path) as f:
        previous = json.load(f)
    stored = {k: v for k, v in previous.get('config', {}).items() if k not in RESUME_IGNORED}
    now = {k: v for k, v in json.loads(json.dumps(config)).items() if k not in RESUME_IGNORED}
    if previous.get('runs') and stored != now:
        changed = sorted(k for k in set(stored) | set(now) if stored.get(k) != now.get(k))
        raise nb.OverwriteError(f'{path}: its runs were made with other settings '
                                f'({", ".join(changed)}); use another --tag, or --overwrite to '
                                f'replace them on purpose')


def resumable(path, labels, variant, args, config):
    """The stored runs that this invocation keeps (ner_baseline.resumable_runs, with the same
    config but for RESUME_IGNORED); none with --overwrite."""
    if args.overwrite or not os.path.exists(path):
        return []
    with open(path) as f:
        previous = json.load(f)
    stored = {k: v for k, v in previous.get('config', {}).items() if k not in RESUME_IGNORED}
    now = {k: v for k, v in json.loads(json.dumps(config)).items() if k not in RESUME_IGNORED}
    if stored != now:
        return []
    return nb.resumable_runs(path, labels, MODEL, variant, args.seed_offset, save_predictions=True)


def select_units(folds, units):
    """The folds whose run_id is one of `units` (a paper's .conll optional), all without them."""
    if not units:
        return folds
    wanted = {u if u in {f[0] for f in folds} else f'{u}.conll' for u in units}
    unknown = sorted(wanted - {f[0] for f in folds})
    if unknown:
        raise SystemExit(f'--units: no unit {unknown}')
    return [f for f in folds if f[0] in wanted]


def run(args):
    out_dir = os.path.join(nb.PROCESSED_DIR, 'ner')
    crf = crf_arguments(args)
    if args.package_trainer:
        check_accelerate()
    data, labels, tie_info = nb.granularity_data(args.granularity, crf)
    names = label_map(args.granularity, labels, coarse_names=args.coarse_names)
    protocol, variant = nb.protocol_name(crf), variant_name(args)
    out_path = results_path(out_dir, args.granularity, protocol, variant, args.max_train)
    config = stored_config(args, crf, names)
    check_resume(out_path, labels, variant, args, config)
    folds = select_units(nb.make_folds(data, crf, indices=True, granularity=args.granularity),
                         args.units)
    device = nb.torch_device(args)
    environment = nb.run_environment(device)
    os.makedirs(out_dir, exist_ok=True)
    results = {'granularity': args.granularity, 'protocol': protocol, 'variant': variant,
               'model': MODEL, 'labels': labels, 'label_tie': tie_info,
               'paper_test_f1': nb.PAPER_F1[args.granularity], 'config': config,
               'runs': resumable(out_path, labels, variant, args, config)}
    done = {r.get('run_id') for r in results['runs']}
    offset = f', seed offset {args.seed_offset}' if args.seed_offset else ''
    print(f'\n=== GLiNER {args.granularity}: {len(data)} sentences, {len(labels)} labels, '
          f'protocol {protocol}, variant {variant}{offset}, device {device} ===')
    for run_id, tr, va, te in folds:
        if run_id in done:
            print(f'  {run_id}: already done, skipping')
            continue
        if args.max_train:
            tr = tr[:args.max_train]
        train, val, test = ([data[i] for i in part] for part in (tr, va, te))
        seed = int(run_id[4:]) if run_id.startswith('seed') else crf.seeds[0]
        seed += args.seed_offset      # as the CRF head's model seed (ner_baseline.run)
        start = time.time()
        if args.zero_shot:
            res = zero_shot(train, val, test, labels, names, device, te)
        elif args.package_trainer:
            epochs = stored_epochs(args.stored_runs, args.granularity, protocol, args.seed_offset,
                                   run_id, seed, names)
            res = package_fine_tune(seed, epochs, train, val, test, labels, names, device, te)
        else:
            res = fine_tune(seed, train, val, test, labels, names, args, device, te)
        res['run_id'] = run_id
        res['runtime_sec'] = round(time.time() - start, 1)
        if environment:
            res['environment'] = environment
        results['runs'].append(res)
        nb.summarize(results)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        chosen = (f"threshold {res['threshold']}" if args.zero_shot
                  else f"{res['epochs']} epochs, the stored best" if args.package_trainer
                  else f"best epoch {res['best_epoch']}")
        print(f"  {run_id}: test F1 {res['f1']:.2f} (strict {res['f1_strict']:.2f}), {chosen}, "
              f"{res['runtime_sec']:.0f}s; running mean {results['test_f1_mean']:.2f}", flush=True)
    if results['runs']:
        nb.summarize(results)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
    print(f'Saved {out_path}')
    return out_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="GLiNER on the CRF head's held-out units "
                                                 '(ROADMAP §4, the week-21 protocol)')
    parser.add_argument('--granularity', choices=['fine', 'coarse', 'mulms', 'sofc'], required=True)
    parser.add_argument('--leave-one-paper-out', action='store_true',
                        help="MaterioMiner (fine, coarse): the CRF head's held-out papers")
    parser.add_argument('--leave-one-subfield-out', action='store_true',
                        help="MuLMS: the CRF head's held-out subfields (with --train-pool)")
    parser.add_argument('--train-pool', type=int, default=0, metavar='N',
                        help='with --leave-one-subfield-out: the pool of N training and '
                             'validation sentences (the CRF head\'s 400)')
    parser.add_argument('--document-folds', type=int, default=0, metavar='K',
                        help="SOFC-Exp: the CRF head's K article folds")
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--zero-shot', action='store_true',
                        help='no training (variant gliner_zs; not for coarse, whose codes have '
                             'no documented expansion)')
    parser.add_argument('--coarse-names', action='store_true',
                        help="E1 (exploratory): MaterioMiner's coarse-grained codes given as their "
                             'class names (COARSE_NAMES), not lowercased; fine-tuning on '
                             '--granularity coarse only (variant gliner_ftn)')
    parser.add_argument('--package-trainer', action='store_true',
                        help="E2 (exploratory): fine-tuning by the package's own trainer "
                             "(gliner.training; it needs accelerate), each unit for the seed's "
                             'best epoch in the stored gliner_ft run (variant gliner_ftp)')
    parser.add_argument('--stored-runs', default=None, metavar='DIR',
                        help='with --package-trainer: where the stored gliner_ft files are '
                             '(default results/gpu)')
    parser.add_argument('--units', nargs='+', default=None, metavar='UNIT',
                        help='run only these units (run ids: a paper, .conll optional; a '
                             'subfield; fold<i>); the results file keeps collecting units; not '
                             'stored in the config')
    parser.add_argument('--tag', type=nb.tag_name, required=True,
                        help='appended to the variant (gliner_ft_m300)')
    parser.add_argument('--seed-offset', type=int, default=0,
                        help='added to the model seed, as for the CRF head (fine-tuning only)')
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'fine-tuning: at most this many epochs (default {EPOCHS})')
    parser.add_argument('--patience', type=int, default=None,
                        help=f'fine-tuning: epochs without a better validation F1 before it '
                             f'stops (default {PATIENCE})')
    parser.add_argument('--grad-accumulation', type=int, default=None, metavar='N',
                        help=f'fine-tuning: each batch of {BATCH_SIZE} sentences in N '
                             f'micro-batches (default 1), if memory requires')
    parser.add_argument('--max-train', type=int, default=0,
                        help='limit training sentences (smoke tests; files suffixed _smoke<N>)')
    parser.add_argument('--overwrite', action='store_true',
                        help='replace the runs of a results file made with other settings')
    parser.add_argument('--device', default=None)
    parser.add_argument('--threads', type=int, default=0)
    args = parser.parse_args(argv)
    protocols = {'fine': 'leave_one_paper_out', 'coarse': 'leave_one_paper_out',
                 'mulms': 'leave_one_subfield_out', 'sofc': 'document_folds'}
    given = [p for p in ('leave_one_paper_out', 'leave_one_subfield_out', 'document_folds')
             if getattr(args, p)]
    if given != [protocols[args.granularity]]:
        parser.error(f"--granularity {args.granularity} runs on the CRF head's "
                     f"--{protocols[args.granularity].replace('_', '-')} units only")
    if args.granularity == 'mulms' and args.train_pool < 1:
        parser.error('--leave-one-subfield-out needs --train-pool N (the CRF head\'s pool)')
    if args.granularity != 'mulms' and args.train_pool:
        parser.error('--train-pool is for --leave-one-subfield-out')
    if args.granularity == 'sofc' and args.document_folds < 2:
        parser.error('--document-folds: at least 2 folds')
    training = ('epochs', 'patience', 'grad_accumulation')
    if args.coarse_names and (args.granularity != 'coarse' or args.zero_shot
                              or args.package_trainer):
        parser.error('--coarse-names (E1) is fine-tuning by this module\'s loop on --granularity '
                     'coarse only')
    if args.package_trainer and args.zero_shot:
        parser.error('--package-trainer (E2) is fine-tuning, not --zero-shot')
    if args.stored_runs is not None and not args.package_trainer:
        parser.error('--stored-runs is for --package-trainer')
    if args.zero_shot:
        if args.granularity not in ZERO_SHOT_GRANULARITIES:
            parser.error(f'--zero-shot: no zero-shot arm for {args.granularity} (its codes have '
                         f'no documented expansion)')
        if args.seed_offset:
            parser.error('--zero-shot is one deterministic run per unit: no --seed-offset')
        for option in training:
            if getattr(args, option) is not None:
                parser.error(f"--{option.replace('_', '-')} is for fine-tuning, not --zero-shot")
    elif args.package_trainer:
        for option in training:
            if getattr(args, option) is not None:
                parser.error(f"--{option.replace('_', '-')} is not for --package-trainer (each "
                             f"unit trains its stored best epoch, in batches of {BATCH_SIZE})")
        if args.stored_runs is None:
            args.stored_runs = str(STORED_RUNS)
    else:
        defaults = {'epochs': EPOCHS, 'patience': PATIENCE, 'grad_accumulation': 1}
        for option, default in defaults.items():
            if getattr(args, option) is None:
                setattr(args, option, default)
        if args.epochs < 1 or args.patience < 1:
            parser.error('--epochs and --patience: at least 1')
        if args.grad_accumulation < 1 or BATCH_SIZE % args.grad_accumulation:
            parser.error(f'--grad-accumulation: a divisor of the batch size {BATCH_SIZE}')
    if args.seed_offset < 0 or args.max_train < 0:
        parser.error('--seed-offset and --max-train: not negative')
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except nb.OverwriteError as e:
        sys.exit(f'error: {e}')


if __name__ == '__main__':
    main()
