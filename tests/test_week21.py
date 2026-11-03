"""Week 21 (ROADMAP §4, the week-21 protocol): GLiNER on the CRF head's held-out units
(gliner_baseline.py): the labels, the conversion of spans to IOB2 tags, the splits against the
stored runs, the parser and file names, batches w21s and w21a (with every earlier batch
unchanged and the runner's new steps), the week-21 sections of ner_seed_analysis.py and
ner_mentions.py dry-run on fake families, and a tiny CPU run of the real model. The labels,
conversions, parser and batches need neither torch nor gliner; the splits need the corpora, the
analysis scipy, and the CPU run gliner 0.2.30 with the pinned model in the Hugging Face cache
(each skipped without them)."""
import copy
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

import decision_tests as dt
import gliner_baseline as gb
import kaggle_jobs as kj
import metrics
import ner_baseline as nb
import ner_mentions as nm
import ner_seed_analysis as sa
from ner_ensemble import tags_to_spans

REPO = Path(__file__).resolve().parents[1]
GPU = REPO / 'results' / 'gpu'
SETTINGS = {'fine': ['--granularity', 'fine', '--leave-one-paper-out'],
            'coarse': ['--granularity', 'coarse', '--leave-one-paper-out'],
            'mulms': ['--granularity', 'mulms', '--leave-one-subfield-out', '--train-pool', '400'],
            'sofc': ['--granularity', 'sofc', '--document-folds', '5']}
PREFIX = {s: sa.WEEK21_SETTINGS[s][0] for s in SETTINGS}
PAPER1 = '10.1007_s11340-021-00758-x.conll'
FINE_LABELS = {
    'Alloy': 'alloy', 'Crack': 'crack', 'CrackInitation': 'crack initiation',
    'DataSet': 'data set', 'FatigueLifetime': 'fatigue lifetime', 'FatigueTest': 'fatigue test',
    'FatigueTestSpecimen': 'fatigue test specimen', 'GrainBoundary': 'grain boundary',
    'LoadingType': 'loading type', 'Location': 'location', 'MaterialClass': 'material class',
    'MechanicalProperty': 'mechanical property',
    'MechanicalTestDataSet': 'mechanical test data set',
    'MechanicalTreatment': 'mechanical treatment', 'Mechanism': 'mechanism',
    'Microstructure': 'microstructure', 'MicrostructureEntity': 'microstructure entity',
    'StateDescriptor': 'state descriptor', 'StrainRate': 'strain rate',
    'StressAmplitude': 'stress amplitude', 'TensileUltimateStrength': 'tensile ultimate strength',
    'TensileYieldStrength': 'tensile yield strength', 'TestSpecimen': 'test specimen',
    'TestingMachine': 'testing machine', 'TestingMedium': 'testing medium',
    'TwinBoundary': 'twin boundary', 'Value': 'value'}
# the protocol's disclosure: gold test entities wider than max_width (12 words), of all
MAX_WIDTH_MISSES = {'fine': (0, 1328), 'coarse': (1, 2194), 'sofc': (4, 4590), 'mulms': (78, 31952)}


def _stored(name):
    return json.loads((GPU / name).read_text())


def _have_mulms():
    return os.path.isfile(os.path.join(nb.MULMS_DIR, 'subfields.json'))


# --- the labels --------------------------------------------------------------------------------

def test_the_labels_are_the_protocols_one_to_one():
    """MaterioMiner fine: the class names camel case split and lowercased, CrackInitation spelled
    out; coarse: the codes lowercased; SOFC-Exp's and MuLMS's tables; in the corpus's order, each
    back to its code one to one (the stored CRF files' labels when they are here)."""
    fine = list(FINE_LABELS)
    assert gb.label_map('fine', fine) == FINE_LABELS
    assert gb.camel_case_label('StressAmplitude') == 'stress amplitude'
    assert gb.camel_case_label('CrackInitation') == 'crack initiation'
    coarse = ['CEL', 'CGE', 'MAT', 'VAL']
    assert gb.label_map('coarse', coarse) == {'CEL': 'cel', 'CGE': 'cge', 'MAT': 'mat',
                                              'VAL': 'val'}
    sofc = ['DEVICE', 'EXPERIMENT', 'MATERIAL', 'VALUE']
    assert list(gb.label_map('sofc', sofc).values()) == ['device', 'experiment', 'material',
                                                         'value']
    mulms = ['CITE', 'DEV', 'FORM', 'INSTRUMENT', 'MAT', 'MEASUREMENT', 'NUM', 'PROPERTY', 'RANGE',
             'SAMPLE', 'TECHNIQUE', 'UNIT', 'VALUE']
    assert list(gb.label_map('mulms', mulms).values()) == [
        'citation', 'device', 'form', 'instrument', 'material', 'measurement', 'number', 'property',
        'range', 'sample', 'technique', 'unit', 'value']
    for granularity, codes in (('fine', fine), ('coarse', coarse), ('sofc', sofc),
                               ('mulms', mulms)):
        names = gb.label_map(granularity, codes)
        assert list(names) == codes                          # the corpus's order
        assert len(set(names.values())) == len(codes)         # one to one
    with pytest.raises(ValueError, match='not one to one'):
        gb.label_map('fine', ['DataSet', 'DATASet'])          # both 'data set'
    with pytest.raises(ValueError, match='label table'):
        gb.label_map('mulms', mulms[:-1])
    with pytest.raises(ValueError, match='label table'):
        gb.label_map('sofc', [*sofc, 'OTHER'])
    if not (GPU / f"ner_results_{PREFIX['fine']}_crf_m0.json").exists():
        pytest.skip('the stored results are not here')
    for setting in SETTINGS:
        labels = _stored(f'ner_results_{PREFIX[setting]}_crf_m0.json')['labels']
        names = gb.label_map(setting, labels)
        assert list(names) == labels and len(set(names.values())) == len(labels)
    assert gb.label_map('fine', _stored(f"ner_results_{PREFIX['fine']}_crf_m0.json")['labels']) \
        == FINE_LABELS


# --- spans and tags ----------------------------------------------------------------------------

NAMES = {'MAT': 'material', 'VAL': 'value'}
CODES_OF = {v: k for k, v in NAMES.items()}


def test_spans_to_iob2_reproduces_the_gold_tags_on_gold_spans():
    """gold_spans (GLiNER's training spans, both ends inclusive, labels) and spans_to_iob2 (its
    decoded spans back to IOB2 tags on the same words) reproduce any IOB2 tags: adjacent
    entities of one type, an entity at either end, none at all; overlaps are refused."""
    cases = [['B-MAT', 'I-MAT', 'B-MAT', 'O', 'B-VAL'], ['O', 'O', 'O'],
             ['B-VAL', 'B-VAL', 'I-VAL', 'I-VAL'], ['O', 'B-MAT'], ['B-MAT']]
    for tags in cases:
        spans = gb.gold_spans(tags, NAMES)
        assert all(0 <= s <= e < len(tags) and label in CODES_OF for s, e, label in spans)
        assert gb.spans_to_iob2(spans, len(tags), CODES_OF) == tags
    assert gb.gold_spans(['B-MAT', 'I-MAT', 'B-MAT', 'O', 'B-VAL'], NAMES) == [
        (0, 1, 'material'), (2, 2, 'material'), (4, 4, 'value')]
    # an I- tag that continues nothing starts an entity, as seqeval's default mode reads it
    assert gb.gold_spans(['O', 'I-VAL', 'I-VAL'], NAMES) == [(1, 2, 'value')]
    with pytest.raises(ValueError, match='overlaps'):
        gb.spans_to_iob2([(0, 2, 'material'), (2, 3, 'value')], 4, CODES_OF)
    with pytest.raises(ValueError, match='outside'):
        gb.spans_to_iob2([(3, 4, 'material')], 4, CODES_OF)
    with pytest.raises(KeyError):
        gb.spans_to_iob2([(0, 0, 'person')], 2, CODES_OF)


def test_a_decode_filtered_by_score_is_the_decode_at_the_higher_threshold():
    """`above` keeps the scores over the threshold as the package compares them, in float32: a
    score equal to float32(t) is out at t, as torch's probs > t has it."""
    t = gb.float32(0.4)
    spans = [(0, 0, 'material', gb.float32(0.3)), (1, 2, 'value', t), (3, 3, 'value', 0.41)]
    assert gb.above(spans, 0.4) == [spans[2]]
    assert gb.above(spans, 0.3) == spans[1:]
    assert gb.predicted_tags([spans], [['w'] * 4], CODES_OF, 0.4) == [['O', 'O', 'O', 'B-VAL']]


def test_the_packages_flat_decoding_filtered_from_its_lowest_threshold():
    """GLiNER's own decoder: one flat decode at 0.3 filtered by score equals its decode at 0.4,
    0.5 and 0.6, on random scores with overlapping candidates (the zero-shot grid's shortcut)."""
    torch = pytest.importorskip('torch')
    decoder_module = pytest.importorskip('gliner.decoding.decoder')
    config = type('Config', (), {})()
    decoder = decoder_module.SpanDecoder(config)
    generator = torch.Generator().manual_seed(0)
    tokens = [['w'] * 9, ['w'] * 6]
    id_to_classes = {1: 'material', 2: 'value'}
    for _ in range(20):
        logits = torch.randn(2, 9, 4, 2, generator=generator) * 2
        low = decoder.decode(tokens, id_to_classes, logits, flat_ner=True, threshold=0.3,
                             multi_label=False)
        for t in (0.4, 0.5, 0.6):
            direct = decoder.decode(tokens, id_to_classes, logits, flat_ner=True, threshold=t,
                                    multi_label=False)
            as_tuples = [[(s.start, s.end, s.entity_type, s.score) for s in row] for row in direct]
            assert [gb.above([(s.start, s.end, s.entity_type, s.score) for s in row], t)
                    for row in low] == as_tuples


def test_the_conversion_on_the_corpora_and_the_width_disclosure():
    """On every sentence of each corpus here the conversion reproduces the gold tags, and the gold
    test entities wider than max_width are the protocol's 0, 1 of 2,194, 4 of 4,590 and 78 of
    31,952. Only tags and counts are read."""
    for setting, argv in SETTINGS.items():
        if setting == 'mulms' and not _have_mulms():
            continue
        args = gb.parse_args([*argv, '--tag', 'm0'])
        crf = gb.crf_arguments(args)
        data, labels, _ = nb.granularity_data(setting, crf)
        names = gb.label_map(setting, labels)
        codes_of = {v: k for k, v in names.items()}
        for words, tags, _ in data:
            assert gb.spans_to_iob2(gb.gold_spans(tags, names), len(words), codes_of) == list(tags)
        spans = [s for _, _, te in (f[1:] for f in nb.make_folds(data, crf, indices=True,
                                                                  granularity=setting))
                 for i in te for s in tags_to_spans(data[i][1])[0]]
        wide = sum(end - start > gb.MAX_WIDTH for start, end, _ in spans)
        assert (wide, len(spans)) == MAX_WIDTH_MISSES[setting]


# --- the splits --------------------------------------------------------------------------------

def test_the_splits_are_the_stored_runs():
    """Every unit's test positions are the stored CRF run's test_index, its training and
    validation sentences the stored numbers and the positions that the protocol's split gives
    (ner_mentions.run_split, from the stored file's config), and the config's protocol keys are
    the stored CRF file's (the corpora must be here; MuLMS is built locally)."""
    if not (GPU / f"ner_results_{PREFIX['fine']}_crf_m0.json").exists():
        pytest.skip('the stored results are not here')
    # the protocol's sizes: (least, most) training and validation sentences per unit
    sizes = {'fine': ((234, 370), (41, 65)), 'coarse': ((234, 370), (41, 65)),
             'sofc': ((534, 628), (94, 111)), 'mulms': ((340, 340), (60, 60))}
    checked = 0
    for setting, argv in SETTINGS.items():
        if setting == 'mulms' and not _have_mulms():
            continue
        args = gb.parse_args([*argv, '--tag', 'm0'])
        crf = gb.crf_arguments(args)
        data, labels, _ = nb.granularity_data(setting, crf)
        folds = nb.make_folds(data, crf, indices=True, granularity=setting)
        stored = _stored(f'ner_results_{PREFIX[setting]}_crf_m0.json')
        assert labels == stored['labels']
        assert [f[0] for f in folds] == [r['run_id'] for r in stored['runs']]
        config = gb.stored_config(args, crf, gb.label_map(setting, labels))
        assert {k: config[k] for k in gb.PROTOCOL_KEYS if k in config} == {
            k: stored['config'][k] for k in gb.PROTOCOL_KEYS if k in stored['config']}
        for (run_id, tr, va, te), run in zip(folds, stored['runs'], strict=True):
            assert te == run['test_index']
            assert (len(tr), len(va), len(te)) == (run['n_train'], run['n_val'], run['n_test'])
            _, train, test = nm.run_split(stored, run, data, setting)
            assert (train, test) == (tr, te)
            assert not set(tr) & set(va) and not (set(tr) | set(va)) & set(te)
            checked += 1
        n_train, n_val = ([len(f[k]) for f in folds] for k in (1, 2))
        assert ((min(n_train), max(n_train)), (min(n_val), max(n_val))) == sizes[setting]
    assert checked >= 13


# --- the parser and the files ------------------------------------------------------------------

def test_the_parser_names_and_pins(capsys):
    args = gb.parse_args([*SETTINGS['mulms'], '--tag', 'm300', '--seed-offset', '300'])
    assert (gb.variant_name(args), args.epochs, args.patience, args.grad_accumulation) == (
        'gliner_ft_m300', 20, 3, 1)
    crf = gb.crf_arguments(args)
    assert nb.protocol_name(crf) == 'leave_one_subfield_out' and crf.train_pool == 400
    path = gb.results_path('out', 'mulms', nb.protocol_name(crf), gb.variant_name(args))
    assert path == os.path.join('out',
                                'ner_results_mulms_leave_one_subfield_out_gliner_ft_m300.json')
    assert gb.results_path('out', 'fine', 'leave_one_paper_out', 'gliner_zs_m0', 2).endswith(
        'ner_results_fine_leave_one_paper_out_gliner_zs_m0_smoke2.json')
    # collect copies the real files and skips the smoke ones, as for every family
    assert nb.parse_result_name(os.path.basename(path)) == (
        'mulms', 'leave_one_subfield_out', 'gliner_ft_m300')
    zs = gb.parse_args([*SETTINGS['sofc'], '--zero-shot', '--tag', 'm0'])
    assert gb.variant_name(zs) == 'gliner_zs_m0' and zs.epochs is None
    config = gb.stored_config(zs, gb.crf_arguments(zs), gb.label_map('sofc', list(gb.SOFC_LABELS)))
    assert config['thresholds'] == [0.3, 0.4, 0.5, 0.6] and 'epochs' not in config
    assert (config['gliner_version'], config['model_revision'], config['backbone_revision']) == (
        '0.2.30', '40ec419335d09393f298636f471328b722c6da9e',
        '8ccc9b6f36199bec6961081d44eb72fb3f7353f3')
    assert config['document_folds'] == 5 and config['label_names']['DEVICE'] == 'device'
    ft = gb.stored_config(args, crf, gb.label_map('mulms', list(gb.MULMS_LABELS)))
    assert (ft['lr_encoder'], ft['lr_others'], ft['weight_decay'], ft['warmup_ratio'],
            ft['batch_size'], ft['epochs'], ft['patience'], ft['threshold'], ft['dtype']) == (
        1e-5, 5e-5, 0.01, 0.1, 8, 20, 3, 0.5, 'float32')
    assert (ft['max_width'], ft['max_len'], ft['flat_ner'], ft['multi_label']) == (12, 384, True,
                                                                                    False)
    # the pins are kaggle_jobs.py's
    assert kj.GLINER_PINS['models'] == {gb.MODEL: gb.MODEL_REVISION,
                                        gb.BACKBONE: gb.BACKBONE_REVISION}
    assert kj.GLINER_PINS['pip_no_deps'] == (f'gliner=={gb.GLINER_VERSION}',)
    for argv, error in (
            ([*SETTINGS['fine'][:2], '--tag', 'm0'], 'leave-one-paper-out units only'),
            (['--granularity', 'fine', '--document-folds', '5', '--tag', 'm0'], 'units only'),
            (['--granularity', 'mulms', '--leave-one-subfield-out', '--tag', 'm0'],
             'needs --train-pool'),
            ([*SETTINGS['sofc'][:2], '--document-folds', '1', '--tag', 'm0'], 'at least 2'),
            ([*SETTINGS['coarse'], '--zero-shot', '--tag', 'm0'], 'no zero-shot arm'),
            ([*SETTINGS['fine'], '--zero-shot', '--tag', 'm0', '--seed-offset', '100'],
             'no --seed-offset'),
            ([*SETTINGS['fine'], '--zero-shot', '--tag', 'm0', '--epochs', '2'], 'fine-tuning'),
            ([*SETTINGS['fine'], '--tag', 'm0', '--grad-accumulation', '3'], 'divisor'),
            ([*SETTINGS['fine'], '--tag', 'M0'], 'invalid tag'),
            (SETTINGS['fine'], 'required')):
        with pytest.raises(SystemExit):
            gb.parse_args(argv)
        assert error in capsys.readouterr().err
    folds = [(PAPER1, [], [], []), ('10.1038_ncomms7164.conll', [], [], [])]
    assert gb.select_units(folds, ['10.1007_s11340-021-00758-x']) == folds[:1]
    assert gb.select_units(folds, None) == folds
    with pytest.raises(SystemExit, match='no unit'):
        gb.select_units(folds, ['fold9'])


def test_no_listing_of_ner_baseline_reads_the_gliner_files(tmp_path, capsys):
    """ner_baseline --compare pairs every variant file of a directory with its reference; a GLiNER
    file is none of its variants (it would be set against plain MatSciBERT), so it is left out."""
    def results(f1):
        return {'runs': [{'run_id': PAPER1, 'f1': f1, 'f1_strict': f1,
                          'per_class': {'macro avg': {'f1-score': f1 / 100},
                                        'weighted avg': {'f1-score': f1 / 100}}}]}
    prefix = 'ner_results_fine_leave_one_paper_out'
    for name, f1 in (('', 40.0), ('_crf', 45.0), ('_gliner_ft_m0', 50.0),
                     ('_gliner_zs_m0', 30.0)):
        _write(tmp_path / f'{prefix}{name}.json', results(f1))
    out = nb.compare_all(str(tmp_path))
    assert list(out) == [f'{prefix}_crf'] and out[f'{prefix}_crf']['reference'] == 'baseline'
    assert 'gliner' not in capsys.readouterr().out


def test_resume_needs_the_same_settings(tmp_path):
    """A results file is resumed only with the same labels, model, variant, seed offset and
    config (but the device and threads), as ner_baseline.py resumes; otherwise the run stops
    before any training, unless --overwrite."""
    args = gb.parse_args([*SETTINGS['fine'], '--tag', 'm0', '--device', 'cpu'])
    config = gb.stored_config(args, gb.crf_arguments(args), FINE_LABELS)
    labels = list(FINE_LABELS)
    path = tmp_path / 'f.json'
    run = {'run_id': PAPER1, 'f1': 50.0, 'test_pred': ['O']}
    path.write_text(json.dumps({'labels': labels, 'model': gb.MODEL, 'variant': 'gliner_ft_m0',
                                'config': {**config, 'threads': 2, 'device': 'cuda'},
                                'runs': [run]}))
    gb.check_resume(str(path), labels, 'gliner_ft_m0', args, config)
    assert gb.resumable(str(path), labels, 'gliner_ft_m0', args, config) == [run]
    other = gb.parse_args([*SETTINGS['fine'], '--tag', 'm0', '--grad-accumulation', '2'])
    changed = gb.stored_config(other, gb.crf_arguments(other), FINE_LABELS)
    with pytest.raises(nb.OverwriteError, match='grad_accumulation'):
        gb.check_resume(str(path), labels, 'gliner_ft_m0', other, changed)
    assert gb.resumable(str(path), labels, 'gliner_ft_m0', other, changed) == []
    with pytest.raises(nb.OverwriteError, match='labels'):
        gb.check_resume(str(path), labels[::-1], 'gliner_ft_m0', args, config)
    forced = gb.parse_args([*SETTINGS['fine'], '--tag', 'm0', '--grad-accumulation', '2',
                            '--overwrite'])
    gb.check_resume(str(path), labels, 'gliner_ft_m0', forced, changed)
    assert gb.resumable(str(path), labels, 'gliner_ft_m0', forced, config) == []


def test_the_optimizer_groups_are_the_package_trainers():
    """The encoder (token_rep_layer) at 1e-5 and the rest at 5e-5, weight decay 0.01 but on the
    biases and the LayerNorm weights (the package Trainer's rule)."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.token_rep_layer = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.LayerNorm(2))
            self.span_rep_layer = torch.nn.Linear(2, 2)

    toy = torch.nn.Module()
    toy.model = Toy()
    groups = gb.parameter_groups(toy)
    named = {id(p): n for n, p in toy.named_parameters()}
    got = sorted((g['lr'], g['weight_decay'], tuple(sorted(named[id(p)] for p in g['params'])))
                 for g in groups)
    assert got == sorted([
        (5e-5, 0.01, ('model.span_rep_layer.weight',)),
        (5e-5, 0.0, ('model.span_rep_layer.bias',)),
        (1e-5, 0.01, ('model.token_rep_layer.0.weight',)),
        (1e-5, 0.0, ('model.token_rep_layer.0.bias', 'model.token_rep_layer.1.bias',
                     'model.token_rep_layer.1.weight'))])


# --- batches w21s and w21a ---------------------------------------------------------------------

def _stub_inputs(monkeypatch):
    monkeypatch.setattr(kj, 'corpus_sha256',
                        lambda report=kj.CORPUS_REPORT: 'sha:' + Path(report).name)
    monkeypatch.setattr(kj, 'mulms_files', lambda report=None: {'a_01.conll': 'sha:mulms'})
    monkeypatch.setattr(kj, 'file_sha256', lambda path: 'sha:' + Path(path).name)


# the batch specs of every batch up to week 20 (BATCHES through w20a and stage-2 names), their
# external inputs stubbed, as JSON: its SHA-256 before week 21's code (commit d227126)
EARLIER_SPECS_SHA256 = '393cb2da102cc771c74c2ab80162f18fb7ded7b9cca2c53e4562b9e8215d02d0'


def test_every_earlier_batch_is_unchanged(monkeypatch):
    _stub_inputs(monkeypatch)
    names = list(kj.BATCHES[:kj.BATCHES.index('w20a') + 1])
    names += [f'w6c:{m}' for m in kj.STAGE2] + ['w6c:deberta,ctx', 'w6c:deberta,mtst',
                                               'w6c:ctx,sofc-ctx']
    specs = {n: kj.batch_spec(n) for n in names}
    assert len(specs) == 68
    text = json.dumps(specs, sort_keys=True)
    assert hashlib.sha256(text.encode()).hexdigest() == EARLIER_SPECS_SHA256
    assert not any('pip_no_deps' in s for s in specs.values())
    assert not any(j[0] == kj.GLINER_SCRIPT for s in specs.values() for lane in s['lanes']
                   for j in lane)


def _run_jobs():
    spec = importlib.util.spec_from_file_location('run_jobs',
                                                  REPO / 'src' / 'kaggle' / 'run_jobs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_runner_runs_gliner_jobs_and_installs_gliner_alone():
    rj = _run_jobs()
    assert rj.job_command(['--crf'])[1:] == ['src/graph/ner_baseline.py', '--crf']
    job = kj.gliner_job('fine')
    assert rj.job_command(job)[1:] == job and rj.SCRIPTS == (kj.GLINER_SCRIPT,)
    plain = {'pip': ['transformers==5.18.0']}
    assert rj.pip_commands(plain, ['w.whl']) == {
        'pip': [sys.executable, '-m', 'pip', 'install', '-q', 'transformers==5.18.0', 'w.whl']}
    commands = rj.pip_commands({**plain, 'pip_no_deps': ['gliner==0.2.30']}, ['w.whl'])
    assert list(commands) == ['pip', 'pip_no_deps']
    assert commands['pip'] == rj.pip_commands(plain, ['w.whl'])['pip']
    assert commands['pip_no_deps'] == [sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
                                       'gliner==0.2.30']


def test_the_runner_installs_gliner_after_the_pins(tmp_path, monkeypatch):
    """run_jobs.main: a spec without pip_no_deps makes the one install of before; with it, a
    second install of those packages alone, both recorded in the manifest (the setup then stops
    here at the missing code)."""
    rj = _run_jobs()
    monkeypatch.setattr(rj, 'WORK', tmp_path / 'working')
    monkeypatch.setattr(rj, 'INPUT', tmp_path / 'input')
    (tmp_path / 'input').mkdir()
    (tmp_path / 'input' / 'w.whl').write_text('')
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out = 'Tesla T4\nTesla T4\n' if cmd[0] == 'nvidia-smi' else ''
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr='')
    monkeypatch.setattr(rj.subprocess, 'run', fake_run)
    base = {'gpu': 'Tesla T4', 'pip': ['transformers==5.18.0'], 'wheels': ['w.whl'],
            'lanes': [[], []], 'models': {}, 'corpus_sha256': 'x' * 64}
    for spec, keys in ((base, ['pip']),
                       ({**base, 'pip_no_deps': ['gliner==0.2.30']}, ['pip', 'pip_no_deps'])):
        calls.clear()
        rj.main(spec)
        setup = json.loads((tmp_path / 'working' / 'manifest.json').read_text())['setup']
        assert [k for k in ('pip', 'pip_no_deps') if k in setup] == keys
        assert all(setup[k] == 0 and f'{k}_output' in setup for k in keys)
        installs = [c for c in calls if c[1:4] == ['-m', 'pip', 'install']]
        assert installs == list(rj.pip_commands(spec, [tmp_path / 'input' / 'w.whl']).values())
        assert 'code.tar.gz' in setup['error']


def _gliner_args(job):
    assert job[0] == kj.GLINER_SCRIPT
    return gb.parse_args(job[1:])


def _written(job):
    """(granularity, protocol, variant) of the file a job writes."""
    if job[0] == kj.GLINER_SCRIPT:
        args = _gliner_args(job)
        return args.granularity, nb.protocol_name(gb.crf_arguments(args)), gb.variant_name(args)
    args = nb.parse_args(job)
    return args.granularity, nb.protocol_name(args), nb.variant_name(args)


def test_week21_batches(monkeypatch):
    """w21s: smoke4's CRF repeat (smokebl), the timed fine-tuning runs at seed offset 0 on one unit
    per corpus (MaterioMiner fine paper 1, MuLMS's first subfield, SOFC-Exp's fold 0), the
    determinism repeat of the paper-1 run (smokebm, the other lane) and zero-shot on paper 1.
    w21a: smoke4's repeat (smokebn), fine-tuning at seed offsets 0 to 900 on the four settings and
    zero-shot on fine, MuLMS and SOFC-Exp, each job its own file. Both carry the GLiNER pins."""
    _stub_inputs(monkeypatch)
    smokebh = kj.batch_lanes('w20a')[0][0]
    lane0, lane1 = kj.batch_lanes('w21s')
    assert lane0[0] == smokebh[:-1] + ['smokebl']
    timed = [_gliner_args(j) for j in (lane0[1], lane0[2], lane1[1])]
    assert [(a.granularity, a.units, a.tag, a.zero_shot) for a in timed] == [
        ('fine', [kj.PAPERS[0]], 'm0', False), ('mulms', ['electrolysis'], 'm0', False),
        ('sofc', ['fold0'], 'm0', False)]
    repeat = _gliner_args(lane1[0])
    assert repeat.tag == 'smokebm' and lane1[0] == [a if a != 'm0' else 'smokebm' for a in lane0[1]]
    zero = _gliner_args(lane1[2])
    assert (zero.granularity, zero.units, zero.zero_shot, zero.tag) == ('fine', [kj.PAPERS[0]],
                                                                         True, 'm0')
    assert sa.WEEK21_REPEATS['smokebm'] == (
        os.path.basename(gb.results_path('', 'fine', 'leave_one_paper_out', 'gliner_ft_smokebm')),
        os.path.basename(gb.results_path('', 'fine', 'leave_one_paper_out', 'gliner_ft_m0')))
    for a in timed + [repeat, zero]:
        assert a.seed_offset == 0 and a.threads == 2 and not a.max_train
    files = [_written(j) for j in lane0 + lane1]
    assert len(files) == len(set(files))
    # w21a: every arm, a job per setting and offset
    a0, a1 = kj.batch_lanes('w21a')
    assert a0[0] == smokebh[:-1] + ['smokebn'] and abs(len(a0) - 1 - len(a1)) <= 1
    jobs = [_gliner_args(j) for j in a0[1:] + a1]
    fine_tuned = sorted((a.granularity, a.seed_offset) for a in jobs if not a.zero_shot)
    assert fine_tuned == sorted((g, off) for g in SETTINGS for off in range(0, 1000, 100))
    assert sorted(a.granularity for a in jobs if a.zero_shot) == ['fine', 'mulms', 'sofc']
    for a in jobs:
        assert a.tag == f'm{a.seed_offset}' and a.units is None and a.threads == 2
        assert (a.epochs, a.patience, a.grad_accumulation) == ((None,) * 3 if a.zero_shot
                                                               else (20, 3, 1))
        crf = gb.crf_arguments(a)
        stored = _stored(f'ner_results_{PREFIX[a.granularity]}_crf_m0.json')['config']
        config = gb.stored_config(a, crf, {})
        assert {k: config[k] for k in gb.PROTOCOL_KEYS if k in config} == {
            k: stored[k] for k in gb.PROTOCOL_KEYS if k in stored}
    files = [_written(j) for j in a0 + a1]
    assert len(files) == len(set(files)) == 1 + 40 + 3
    # the timed runs and zero-shot unit of w21s write files that w21a's jobs extend
    assert {_written(j) for j in (lane0[1], lane0[2], lane1[1], lane1[2])} <= set(files)
    for name in ('w21s', 'w21a'):
        assert name in kj.BATCHES and kj.batch_name(name) == name
        spec = kj.batch_spec(name)
        assert spec['pip'] == list(kj.PIP) + ['sentencepiece==0.2.2']
        assert spec['pip_no_deps'] == ['gliner==0.2.30']
        assert spec['models'] == {**kj.MODELS, gb.MODEL: gb.MODEL_REVISION,
                                  gb.BACKBONE: gb.BACKBONE_REVISION}
        assert spec['mulms_files'] == {'a_01.conll': 'sha:mulms'}
        assert 'topic_corpora' not in spec and 'week19_names' not in spec
        assert 'sofc_corpus_sha256' not in spec


# --- the week-21 section of ner_seed_analysis.py, dry-run on fake families ----------------------

# per setting, the fake fine-tuned and zero-shot micro F1 shifts per unit (in the CRF files' order)
FT_SHIFT = {'fine': [1.0, 2.0, 3.0, 2.5], 'coarse': [-1.0, -2.0, -1.5, -3.0],
            'mulms': [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.0], 'sofc': [0.5, 1.0, 1.5, 0.7, 1.2]}
ZS_SHIFT = {'fine': [-20.0, -25.0, -22.0, -30.0], 'mulms': [-10.0] * 7,
            'sofc': [-40.0, -35.0, -38.0, -41.0, -36.0]}


def _write(path, data):
    path.write_text(json.dumps(data))


def _fake(stored, variant, shifts, threshold=None):
    fake = copy.deepcopy(stored)
    fake['variant'] = variant
    for run, shift in zip(fake['runs'], shifts, strict=True):
        run['f1'] = round(run['f1'] + shift, 2)
        if threshold is not None:
            run.update(threshold=threshold, seed=None, best_epoch=None)
    return fake


@pytest.fixture(scope='module')
def dry21(tmp_path_factory):
    """A results directory with the stored CRF files of the four settings (linked) and fake
    families: gliner_ft_m0 to _m900, copies of the CRF files with micro F1 shifted by FT_SHIFT;
    gliner_zs_m0, a copy of crf_m0 shifted by ZS_SHIFT (fine, MuLMS, SOFC-Exp); and a pulled
    output with the determinism repeat, the fake paper-1 run with another runtime_sec."""
    if not (GPU / f"ner_results_{PREFIX['fine']}_crf_m0.json").exists():
        pytest.skip('the stored results are not here')
    root = tmp_path_factory.mktemp('w21_dry')
    results, pulled = root / 'results', root / 'pulled' / 'processed' / 'ner'
    results.mkdir()
    pulled.mkdir(parents=True)
    for setting, prefix in PREFIX.items():
        for off in range(0, 1000, 100):
            name = f'ner_results_{prefix}_crf_m{off}.json'
            (results / name).symlink_to(GPU / name)
            _write(results / f'ner_results_{prefix}_gliner_ft_m{off}.json',
                   _fake(_stored(name), f'gliner_ft_m{off}', FT_SHIFT[setting]))
        if setting in ZS_SHIFT:
            _write(results / f'ner_results_{prefix}_gliner_zs_m0.json',
                   _fake(_stored(f'ner_results_{prefix}_crf_m0.json'), 'gliner_zs_m0',
                         ZS_SHIFT[setting], threshold=0.4))
    name, stored = sa.WEEK21_REPEATS['smokebm']
    repeat = json.loads((results / stored).read_text())
    repeat['runs'] = [dict(repeat['runs'][0], runtime_sec=1.0)]
    _write(pulled / name, repeat)
    return results, root / 'pulled'


def _unit_means(prefix, system, shifts=None):
    files = [_stored(f'ner_results_{prefix}_{system}_m{off}.json') for off in range(0, 1000, 100)]
    ids = [r['run_id'] for r in files[0]['runs']]
    return {i: statistics.fmean(next(r['f1'] for r in f['runs'] if r['run_id'] == i)
                                for f in files) for i in ids}


def test_the_week21_decisions_on_fake_families(dry21):
    """21.1 to 21.4 on the fake families: the per-unit differences are the shifts, the verdicts
    follow the rule (21.1 and 21.4 pass, 21.2 is below, 21.3 is a null read within 3.5 points),
    the dependence readings are decision_tests.dependent_t's with layout(n), and the checks pass;
    zero-shot is reported against the CRF head, its thresholds per unit."""
    pytest.importorskip('scipy')
    results, pulled = dry21
    w21 = sa.week21(results, pulled)
    assert set(w21) == {'checks', 'repeats', 'decisions', 'corpora', 'zero_shot'}
    checks = w21['checks']
    assert (checks['n_files'], checks['splits_equal'], checks['environment_equal']) == (43, True,
                                                                                        True)
    assert checks['n_runs'] == 10 * (4 + 4 + 7 + 5) + 4 + 7 + 5
    r = w21['repeats']['smokebm']
    assert (r['equal'], r['n_runs'], r['environment_equal'], r['missing']) == (True, 1, True, [])
    assert w21['repeats']['status'] == {'checked': True, 'unchecked': [], 'repeat_differs': False,
                                        'splits_differ': False, 'environment_differs': False}
    verdicts = {'fine': ('21.1', True, False), 'coarse': ('21.2', False, True),
                'mulms': ('21.3', False, False), 'sofc': ('21.4', True, False)}
    for setting, e in w21['decisions'].items():
        decision, passes, below = verdicts[setting]
        assert (e['decision'], e['passes'], e['below'], e['decided']) == (decision, passes, below,
                                                                          True)
        d = e['minus_crf']['micro']
        assert d['per_run'] == FT_SHIFT[setting]
        lo, hi = metrics.mean_ci(FT_SHIFT[setting])['ci']
        assert d['ci95'] == pytest.approx([lo, hi], abs=0.006) and 'flag' not in e
        crf = _unit_means(PREFIX[setting], 'crf')
        assert e['crf']['micro']['per_run'] == [round(v, 2) for v in crf.values()]
        for (unit, both), shift in zip(e['units'].items(), FT_SHIFT[setting]):
            assert both['gliner_ft']['micro'] == pytest.approx(round(crf[unit] + shift, 2),
                                                               abs=0.011)
            assert both['gliner_ft']['macro'] == both['crf']['macro']
        order = list(dt.MULMS_UNITS) if setting == 'mulms' else list(e['units'])
        values = [dict(zip(e['units'], d['per_run']))[u] for u in order]
        dep = dt.dependent_t(values, dt.layout(len(values)))
        assert e['dependence'] == dict(dep, above_0=dep['ci95'][0] > 0)
        assert e['validation']['gliner_ft'] == e['validation']['crf']
    assert w21['decisions']['mulms']['reading'] == 'not shown within about 3.5 points'
    assert w21['decisions']['coarse']['reading'] == 'below the CRF head'
    assert w21['corpora'] == {'materiominer': {'21.1': 'passes', '21.2': 'below'},
                              'mulms': {'21.3': 'null'}, 'sofc_exp': {'21.4': 'passes'}}
    assert set(w21['zero_shot']) == {'fine', 'mulms', 'sofc'}
    for setting, e in w21['zero_shot'].items():
        crf_m0 = {r['run_id']: r['f1'] for r in
                  _stored(f'ner_results_{PREFIX[setting]}_crf_m0.json')['runs']}
        crf = _unit_means(PREFIX[setting], 'crf')
        expected = [round(crf_m0[u] + s, 2) - crf[u] for u, s in zip(crf, ZS_SHIFT[setting])]
        assert e['minus_crf']['micro']['per_run'] == pytest.approx(expected, abs=0.011)
        assert all(u['threshold'] == 0.4 for u in e['units'].values())
        order = list(dt.MULMS_UNITS) if setting == 'mulms' else list(e['units'])
        per_unit = dict(zip(e['units'], e['minus_crf']['micro']['per_run']))
        dep = dt.dependent_t([per_unit[u] for u in order], dt.layout(len(order)))
        assert e['dependence'] == dict(dep, above_0=dep['ci95'][0] > 0)
        assert e['seeds'] == {'gliner_zs': [0], 'crf': list(range(0, 1000, 100))}
    # no earlier section reads the GLiNER files
    assert sa.week12_variants(results) == {} and sa.week20(results) == {}


def test_the_week21_checks_and_their_consequences(dry21, tmp_path):
    """Without the repeat the decisions are flagged (decided None); a repeat that differs, a split
    that is not the CRF head's or another environment make them decide nothing (decided False,
    the numbers reported); a family still running is not read; without GLiNER files there is no
    week21 section."""
    pytest.importorskip('scipy')
    results, pulled = dry21
    alone = sa.week21(results)
    assert 'repeats' not in alone
    for e in alone['decisions'].values():
        assert e['decided'] is None and 'smokebm is not checked' in e['flag']
    with pytest.raises(SystemExit, match='no directory'):
        sa.week21(results, tmp_path / 'nowhere')
    name, stored = sa.WEEK21_REPEATS['smokebm']
    differing = tmp_path / 'differing'
    differing.mkdir()
    repeat = json.loads((pulled / 'processed' / 'ner' / name).read_text())
    repeat['runs'][0]['test_pred'][0] = 'B-Alloy'
    _write(differing / name, repeat)
    w21 = sa.week21(results, differing)
    assert w21['repeats']['smokebm']['differ'] == ['test_pred']
    for e in w21['decisions'].values():
        assert e['decided'] is False and 'determinism repeat differs' in e['flag']
        assert 'minus_crf' in e
    # a fresh directory with one run's split, and another run's environment, changed
    bad = tmp_path / 'bad'
    bad.mkdir()
    for path in results.iterdir():
        (bad / path.name).symlink_to(path.resolve())
    fake = json.loads((results / f"ner_results_{PREFIX['sofc']}_gliner_ft_m300.json").read_text())
    fake['runs'][2]['test_index'] = fake['runs'][2]['test_index'][1:]
    fake['runs'][4]['environment'] = dict(fake['runs'][4]['environment'], torch='2.12.0')
    target = bad / f"ner_results_{PREFIX['sofc']}_gliner_ft_m300.json"
    target.unlink()
    _write(target, fake)
    w21 = sa.week21(bad, pulled)
    checks = w21['checks']
    assert checks['splits_differ'] == [f"{target.name} fold2: test_index"]
    assert checks['environment_differs'] == [f"{target.name} fold4"]
    for e in w21['decisions'].values():
        assert e['decided'] is False and "split is not the CRF head's" in e['flag']
        assert "environment is not the stored runs'" in e['flag']
    # a family still running: a unit missing from one file
    target.unlink()
    fake = json.loads((results / target.name).read_text())
    fake['runs'] = fake['runs'][:-1]
    _write(target, fake)
    assert 'sofc' not in sa.week21(bad, pulled)['decisions']
    empty = tmp_path / 'empty'
    empty.mkdir()
    for setting, prefix in PREFIX.items():
        name = f'ner_results_{prefix}_crf_m0.json'
        (empty / name).symlink_to(GPU / name)
    assert sa.week21(empty) == {} and sa.week21(empty, pulled) == {}


def test_main_prints_the_week21_section(dry21, capsys):
    pytest.importorskip('scipy')
    results, pulled = dry21
    sa.print_week21(sa.week21(results, pulled))
    out = capsys.readouterr().out
    assert "splits the CRF head's; environment the stored runs'" in out
    assert 'the repeat smokebm against' in out and 'identical (1 runs' in out
    assert 'week 21, 21.1 (fine): GLiNER fine-tuned' in out and '(passes)' in out
    assert '21.3 (mulms)' in out and 'not shown within about 3.5 points' in out
    assert 'zero-shot (sofc, reported)' in out and 'post hoc dependence' in out


# --- the week-21 section of ner_mentions.py ----------------------------------------------------

def test_the_seen_and_unseen_recall_of_gliner(tmp_path):
    """ner_mentions' week21: per setting, GLiNER fine-tuned (10 seeds) and zero-shot (one run per
    unit) beside the CRF head, the kinds of week 10 against each unit's CRF training sentences;
    absent without the files. Fake families: copies of the CRF head's files (the fine-tuned seeds
    of its seeds, zero-shot of its seed 0), so the differences are known."""
    if not (GPU / f"ner_results_{PREFIX['fine']}_crf_m0.json").exists():
        pytest.skip('the stored results are not here')
    pytest.importorskip('scipy')
    prefix = PREFIX['fine']
    for off in range(0, 1000, 100):
        name = f'ner_results_{prefix}_crf_m{off}.json'
        (tmp_path / name).symlink_to(GPU / name)
    no_corpus = tmp_path / 'no_corpus.txt'
    assert nm.gliner(str(tmp_path)) == {} and 'week21' not in nm.analyse(str(tmp_path), no_corpus)
    for off in range(0, 1000, 100):
        _write(tmp_path / f'ner_results_{prefix}_gliner_ft_m{off}.json',
               _fake(_stored(f'ner_results_{prefix}_crf_m{off}.json'), f'gliner_ft_m{off}',
                     [0] * 4))
    _write(tmp_path / f'ner_results_{prefix}_gliner_zs_m0.json',
           _fake(_stored(f'ner_results_{prefix}_crf_m0.json'), 'gliner_zs_m0', [0] * 4, 0.5))
    out = nm.gliner(str(tmp_path))
    assert list(out) == ['fine']
    e = out['fine']
    assert e['seeds'] == {'crf': list(range(0, 1000, 100)), 'gliner_ft': list(range(0, 1000, 100)),
                          'gliner_zs': [0]}
    assert e['systems']['gliner_ft'] == e['systems']['crf'] and e['systems']['crf']['n_runs'] == 40
    assert e['systems']['gliner_zs']['n_runs'] == 4
    shares = {s: v['unseen']['share'] for s, v in e['systems'].items()}
    assert len(set(shares.values())) == 1                   # the same names and test entities
    for kind in nm.COMPARED:
        assert e['minus_crf']['gliner_ft'][kind]['recall']['mean_diff'] == 0.0
        assert e['minus_crf']['gliner_zs'][kind]['n_pairs'] == 4
    crf_m0 = nm.family_counts([_stored(f'ner_results_{prefix}_crf_m0.json')], nm.corpus('fine'),
                              'fine')[0]
    for unit, systems in e['units'].items():
        sc = nm.scores(crf_m0[unit])
        assert systems['gliner_zs'] == {'recall_seen': sc['seen']['recall'],
                                        'recall_unseen': sc['unseen']['recall']}
    assert nm.analyse(str(tmp_path), no_corpus)['week21'] == out
    # a family still running is left out
    fake = json.loads((tmp_path / f'ner_results_{prefix}_gliner_zs_m0.json').read_text())
    fake['runs'] = fake['runs'][:2]
    _write(tmp_path / f'ner_results_{prefix}_gliner_zs_m0.json', fake)
    assert 'gliner_zs' not in nm.gliner(str(tmp_path))['fine']['systems']


# --- a tiny CPU run of the real model ----------------------------------------------------------

# a toy corpus of four documents (invented words, not corpus text) with SOFC-Exp's four types
TOY = [(['the', 'anode', 'was', 'nickel', 'at', '800', 'K'],
        ['O', 'B-DEVICE', 'O', 'B-MATERIAL', 'O', 'B-VALUE', 'I-VALUE']),
       (['we', 'measured', 'the', 'cell'], ['O', 'B-EXPERIMENT', 'O', 'B-DEVICE']),
       (['zirconia', 'layers', 'were', 'tested'], ['B-MATERIAL', 'O', 'O', 'B-EXPERIMENT'])]
DRIVER = '''import json, sys
sys.path.insert(0, {src!r})
import ner_baseline as nb
import gliner_baseline as gb
data = [(w, t, d) for w, t, d in json.loads({data!r})]
labels = ['DEVICE', 'EXPERIMENT', 'MATERIAL', 'VALUE']
nb.granularity_data = lambda granularity, args: (data, labels, {{}})
gb.main({argv!r})
'''


def _cached_model():
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return (isinstance(try_to_load_from_cache(gb.MODEL, 'model.safetensors',
                                              revision=gb.MODEL_REVISION), str)
            and isinstance(try_to_load_from_cache(gb.BACKBONE, 'spm.model'), str))


def _drive(tmp_path, *flags):
    data = [(words, tags, f'doc{k}.conll') for k in range(4) for words, tags in TOY]
    argv = ['--granularity', 'sofc', '--document-folds', '2', '--units', 'fold0', '--device', 'cpu',
            '--threads', '2', *flags]
    code = DRIVER.format(src=str(REPO / 'src' / 'graph'), data=json.dumps(data), argv=argv)
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PROCESSED_DIR=str(tmp_path))
    proc = subprocess.run([sys.executable, '-I', '-c', code], env=env, capture_output=True,
                          text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    return proc.stdout


def test_a_tiny_cpu_run_of_the_real_model(tmp_path):
    """gliner 0.2.30 and the pinned model on a toy corpus, offline: fine-tuning on 2 sentences
    for 1 epoch writes the CRF schema with the pins; it resumes; run again under another tag it is
    identical but for runtime_sec; zero-shot chooses its threshold on the validation sentences."""
    gliner = pytest.importorskip('gliner')
    if gliner.__version__ != gb.GLINER_VERSION or not _cached_model():
        pytest.skip(f'gliner {gb.GLINER_VERSION} and the pinned model are not here')
    _drive(tmp_path, '--tag', 'm0', '--max-train', '2', '--epochs', '1')
    path = tmp_path / 'ner' / 'ner_results_sofc_document_folds_gliner_ft_m0_smoke2.json'
    res = json.loads(path.read_text())
    assert list(res) == ['granularity', 'protocol', 'variant', 'model', 'labels', 'label_tie',
                         'paper_test_f1', 'config', 'runs', 'test_f1_mean', 'test_f1_std',
                         'test_f1_strict_mean', 'summary']
    assert (res['variant'], res['model'], res['labels']) == (
        'gliner_ft_m0', gb.MODEL, ['DEVICE', 'EXPERIMENT', 'MATERIAL', 'VALUE'])
    config = res['config']
    assert (config['gliner_version'], config['model_revision'], config['backbone_revision'],
            config['document_folds'], config['epochs'], config['max_train']) == (
        gb.GLINER_VERSION, gb.MODEL_REVISION, gb.BACKBONE_REVISION, 2, 1, 2)
    run, = res['runs']
    assert list(run) == ['f1', 'precision', 'recall', 'f1_strict', 'per_class', 'seed',
                         'best_epoch', 'val_f1', 'n_train', 'n_val', 'n_test', 'history',
                         'test_index', 'test_gold', 'test_pred', 'run_id', 'runtime_sec']
    assert (run['run_id'], run['seed'], run['best_epoch'], run['n_train']) == ('fold0', 0, 0, 2)
    assert len(run['history']) == 1 and run['history'][0]['train_loss'] > 0
    assert len(run['test_pred']) == len(run['test_index']) == run['n_test']
    assert all(len(p.split()) == len(g.split()) for p, g in zip(run['test_pred'], run['test_gold']))
    assert 'already done' in _drive(tmp_path, '--tag', 'm0', '--max-train', '2', '--epochs', '1')
    assert json.loads(path.read_text())['runs'] == res['runs']
    _drive(tmp_path, '--tag', 'again', '--max-train', '2', '--epochs', '1')
    again_path = tmp_path / 'ner' / path.name.replace('_m0_', '_again_')
    again, = json.loads(again_path.read_text())['runs']
    assert {k: v for k, v in again.items() if k != 'runtime_sec'} == {
        k: v for k, v in run.items() if k != 'runtime_sec'}
    _drive(tmp_path, '--zero-shot', '--tag', 'm0')
    zs, = json.loads((tmp_path / 'ner' / 'ner_results_sofc_document_folds_gliner_zs_m0.json'
                      ).read_text())['runs']
    assert [h['threshold'] for h in zs['history']] == [0.3, 0.4, 0.5, 0.6]
    best = max(zs['history'], key=lambda h: (h['val_f1'], h['threshold']))
    assert (zs['threshold'], zs['val_f1'], zs['seed'], zs['best_epoch']) == (
        best['threshold'], best['val_f1'], None, None)
    assert zs['test_index'] == run['test_index'] and zs['n_train'] == 5
