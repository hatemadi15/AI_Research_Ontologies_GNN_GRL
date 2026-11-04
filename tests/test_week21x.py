"""Week 21's two exploratory checks of a referee's points (gliner_baseline.py's E1 and E2; outside
the decision count, reported with no decision): E1's class names of MaterioMiner's coarse codes
(COARSE_NAMES: one to one over the 27 codes, each the paper's tag map's and the ontology's label),
E2's epochs from the stored gliner_ft runs, the parser and the configs of both, batch w21x (with
w21s and w21a unchanged), the week21x section of ner_seed_analysis.py dry-run on fake families,
and, where accelerate 1.15.0, gliner 0.2.30 and the pinned model are here, E2's trainer against
the loop and a tiny CPU run of E2 (each skipped without them)."""
import collections
import copy
import hashlib
import json
import os
import statistics
import subprocess
import sys
import types

import pytest

import gliner_baseline as gb
import kaggle_jobs as kj
import metrics
import ner_baseline as nb
import ner_seed_analysis as sa
from ner_ensemble import tags_to_spans
from test_week21 import (FINE_LABELS, GPU, PAPER1, REPO, _cached_model, _drive, _gliner_args,
                         _run_jobs, _stored, _stub_inputs, _write, _written)

OFFSETS = tuple(range(0, 1000, 100))
CODES = ('CEL', 'CGE', 'CHP', 'CPR', 'DAM', 'DAT', 'DEF', 'EQU', 'ETP', 'LOC', 'MAC', 'MAT', 'MEC',
         'MEP', 'MFP', 'MIC', 'MIP', 'MOD', 'MTP', 'OBJ', 'ODE', 'PQT', 'PRO', 'PRP', 'SDE', 'STD',
         'VAL')
COARSE = 'coarse_leave_one_paper_out'
FINE = 'fine_leave_one_paper_out'
FINE_ARGS = ['--granularity', 'fine', '--leave-one-paper-out']
COARSE_ARGS = ['--granularity', 'coarse', '--leave-one-paper-out']


# --- E1: the class names ----------------------------------------------------------------------

# the release's CG-NER_tags.csv (MaterioMiner v1.0.1, doi:10.24406/fordatis/329.2,
# dataset/coarse_grained_ner/), verbatim: the paper's "mapping of the tags to the ontological
# preferred labels"
CG_NER_TAGS = {
    'MAT': 'Material', 'CEL': 'Chemical element', 'CGE': 'Crystallographic entity',
    'CHP': 'Characterization process', 'CPR': 'Chemical property', 'DAM': 'Damage',
    'DEF': 'Defect', 'MIC': 'Microstructure', 'EQU': 'Equipment',
    'ETP': 'Environmental test parameter', 'MAC': 'Material class', 'MEC': 'Mechanism',
    'MEP': 'Mechanical property', 'MFP': 'Manufacturing process',
    'MIP': 'Microstructural property', 'MTP': 'Mechanical test parameter', 'OBJ': 'Object',
    'DAT': 'Data set', 'PRO': 'Property', 'PRP': 'Process parameter', 'VAL': 'Value',
    'LOC': 'Location', 'SDE': 'Shape descriptor', 'ODE': 'Orientation descriptor',
    'STD': 'State descriptor', 'MOD': 'Model', 'PQT': 'Physical quantity'}
# the paper's Fig. 7b: the annotated instances of each coarse-grained type, its bars' numbers
# (ChemicalProperty's, CPR's, has none)
FIG7B = {'MEC': 280, 'VAL': 165, 'MTP': 159, 'MEP': 158, 'DEF': 147, 'DAM': 123, 'CHP': 105,
         'STD': 105, 'MIC': 104, 'OBJ': 103, 'PQT': 87, 'MAC': 78, 'MAT': 75, 'EQU': 71, 'DAT': 68,
         'MFP': 62, 'PRO': 58, 'LOC': 55, 'CEL': 35, 'ETP': 30, 'MIP': 29, 'PRP': 27, 'SDE': 21,
         'ODE': 16, 'CGE': 14, 'MOD': 12}
DATASET = REPO / 'data' / 'raw' / 'dataset'
MM = 'https://w3id.org/pmd/materials-mechanics-ontology/'


def test_the_coarse_class_names_are_one_to_one_over_the_27_codes():
    """COARSE_NAMES gives each of the 27 coarse codes one name, lowercased words, and no two codes
    the same name; label_map maps the corpus's codes, in their order, to them and back, through
    the conversion of spans to tags too. The protocol's labels (the codes lowercased) stay."""
    assert len(CODES) == 27 and sorted(gb.COARSE_NAMES) == list(CODES)
    names = list(gb.COARSE_NAMES.values())
    assert len(set(names)) == 27
    for name in names:
        assert name == name.lower() and name.split() and all(w.isalpha() for w in name.split())
        assert ' '.join(name.split()) == name
    order = list(reversed(CODES))
    mapping = gb.label_map('coarse', order, coarse_names=True)
    assert list(mapping) == order and mapping == {c: gb.COARSE_NAMES[c] for c in order}
    codes_of = {v: k for k, v in mapping.items()}
    assert len(codes_of) == 27 and all(codes_of[mapping[c]] == c for c in order)
    tags = ['B-MEC', 'I-MEC', 'O', 'B-PRP', 'B-PRO', 'I-PRO', 'B-CPR']
    assert gb.spans_to_iob2(gb.gold_spans(tags, mapping), len(tags), codes_of) == tags
    assert gb.label_map('coarse', order) == {c: c.lower() for c in order}
    with pytest.raises(ValueError, match='label table'):
        gb.label_map('coarse', order[:-1], coarse_names=True)
    with pytest.raises(ValueError, match='label table'):
        gb.label_map('coarse', [*order, 'XYZ'], coarse_names=True)
    with pytest.raises(ValueError, match='coarse level only'):
        gb.label_map('fine', list(FINE_LABELS), coarse_names=True)
    stored = GPU / f'ner_results_{COARSE}_crf_m0.json'
    if stored.exists():
        labels = _stored(stored.name)['labels']
        assert sorted(labels) == list(CODES)
        assert list(gb.label_map('coarse', labels, coarse_names=True)) == labels


def _conll_tags(path):
    """The tag sequences of a CoNLL file's sentences (the last column)."""
    sentences, tags = [], []
    for line in path.read_text(encoding='utf-8').split('\n') + ['']:
        if line.strip():
            tags.append(line.split()[-1])
        elif tags:
            sentences.append(tags)
            tags = []
    return sentences


def test_the_coarse_names_are_the_papers_and_the_ontologys():
    """Each name is the preferred label of its code in the release's tag map, lowercased; the
    entities of each code in the dataset's coarse-grained files number its bar in the paper's Fig.
    7b (2,194 in all); and each name is the rdfs:label, lowercased, of the ontology class it
    names (mm:<the name in camel case>)."""
    assert gb.COARSE_NAMES == {c: label.lower() for c, label in CG_NER_TAGS.items()}
    counts = collections.Counter(code for path in sorted((DATASET / 'coarse_grained_ner').glob(
        '*.conll')) for tags in _conll_tags(path) for _, _, code in tags_to_spans(tags)[0])
    assert sorted(counts) == list(CODES) and sum(counts.values()) == 2194
    assert {c: counts[c] for c in FIG7B} == FIG7B
    rdflib = pytest.importorskip('rdflib')
    graph = rdflib.Graph().parse(DATASET / 'ontologies' / 'ontology.ttl', format='turtle')
    for code, name in gb.COARSE_NAMES.items():
        iri = rdflib.URIRef(MM + ''.join(w.capitalize() for w in name.split()))
        assert (iri, rdflib.RDF.type, rdflib.OWL.Class) in graph, code
        assert [str(o).lower() for o in graph.objects(iri, rdflib.RDFS.label)] == [name], code


# --- E2: the epochs, the trainer's arguments ----------------------------------------------------

# the stored gliner_ft runs of MaterioMiner fine paper 1: each seed offset's best epoch + 1
STORED_EPOCHS = {0: 15, 100: 14, 200: 12, 300: 20, 400: 14, 500: 15, 600: 8, 700: 10, 800: 13,
                 900: 18}


def test_the_e2_epochs_are_the_stored_best_epochs(tmp_path):
    """E2 trains each seed's best epoch of the stored gliner_ft run of paper 1 (its best_epoch is
    the 0-based index of the best validation F1 in its history, so best_epoch + 1 epochs); a
    missing file, unit or seed, a smoke run, other labels and a best_epoch that is not that index
    are refused."""
    name = f'ner_results_{FINE}_gliner_ft_m300.json'
    if not (GPU / name).exists():
        pytest.skip('the stored results are not here')
    for off, epochs in STORED_EPOCHS.items():
        assert gb.stored_epochs(str(GPU), 'fine', 'leave_one_paper_out', off, PAPER1, off,
                                FINE_LABELS) == epochs
        run = next(r for r in _stored(f'ner_results_{FINE}_gliner_ft_m{off}.json')['runs']
                   if r['run_id'] == PAPER1)
        assert (run['seed'], run['best_epoch'] + 1) == (off, epochs)
        assert run['history'][epochs - 1]['val_f1'] == run['val_f1'] == max(
            h['val_f1'] for h in run['history'])
    # an E2 job's seed is the CRF head's, the split seed plus the offset, as gliner_ft's
    args = gb.parse_args([*FINE_ARGS, '--package-trainer', '--units', PAPER1[:-len('.conll')],
                          '--tag', 'm300', '--seed-offset', '300'])
    assert gb.crf_arguments(args).seeds[0] + args.seed_offset == 300
    stored = _stored(name)

    def refused(data, match, unit=PAPER1, seed=300, names=FINE_LABELS):
        if data is not None:
            _write(tmp_path / name, data)
        with pytest.raises(SystemExit, match=match):
            gb.stored_epochs(str(tmp_path), 'fine', 'leave_one_paper_out', 300, unit, seed, names)

    refused(None, 'no stored gliner_ft_m300 file')
    refused(stored, 'no run of fold9 with seed 300', unit='fold9')
    refused(stored, 'with seed 301', seed=301)
    refused(stored, 'with these labels', names={**FINE_LABELS, 'Alloy': 'alloys'})
    smoke = copy.deepcopy(stored)
    smoke['config']['max_train'] = 2
    refused(smoke, 'not a full gliner_ft_m300 run')
    beyond = copy.deepcopy(stored)
    run = next(r for r in beyond['runs'] if r['run_id'] == PAPER1)
    run['best_epoch'] = len(run['history'])
    refused(beyond, 'is not the index')
    _write(tmp_path / name, stored)
    assert gb.stored_epochs(str(tmp_path), 'fine', 'leave_one_paper_out', 300, PAPER1, 300,
                            FINE_LABELS) == STORED_EPOCHS[300]


def test_the_parser_and_the_configs_of_e1_and_e2(capsys):
    """E1 (gliner_ftn) is gliner_ft's coarse fine-tuning whose config differs in the label names
    only; E2 (gliner_ftp) records the package's trainer, the accelerate version, the stored runs
    its epochs come from and the trainer's arguments, gliner_ft's recipe under the package's names
    (the schedule over gliner_ft's 20 epochs), and shares gliner_ft's other keys; the options that
    do not apply are refused."""
    e1 = gb.parse_args([*COARSE_ARGS, '--coarse-names', '--tag', 'm300', '--seed-offset', '300'])
    ft = gb.parse_args([*COARSE_ARGS, '--tag', 'm300', '--seed-offset', '300'])
    assert (gb.variant_name(e1), gb.variant_name(ft)) == ('gliner_ftn_m300', 'gliner_ft_m300')
    assert (e1.epochs, e1.patience, e1.grad_accumulation, e1.stored_runs) == (20, 3, 1, None)
    names = gb.label_map('coarse', list(CODES), coarse_names=True)
    c1 = gb.stored_config(e1, gb.crf_arguments(e1), names)
    c0 = gb.stored_config(ft, gb.crf_arguments(ft), gb.label_map('coarse', list(CODES)))
    assert list(c1) == list(c0) and [k for k in c1 if c1[k] != c0[k]] == ['label_names']
    assert c1['label_names'] == names and c1['zero_shot'] is False
    path = gb.results_path('out', 'coarse', 'leave_one_paper_out', gb.variant_name(e1))
    assert nb.parse_result_name(os.path.basename(path)) == ('coarse', 'leave_one_paper_out',
                                                           'gliner_ftn_m300')
    e2 = gb.parse_args([*FINE_ARGS, '--package-trainer', '--units', '10.1007_s11340-021-00758-x',
                        '--tag', 'm300', '--seed-offset', '300'])
    assert gb.variant_name(e2) == 'gliner_ftp_m300'
    assert (e2.epochs, e2.patience, e2.grad_accumulation) == (None, None, None)
    assert e2.stored_runs == str(gb.STORED_RUNS) == str(REPO / 'results' / 'gpu')
    c2 = gb.stored_config(e2, gb.crf_arguments(e2), FINE_LABELS)
    assert (c2['trainer'], c2['accelerate_version'], c2['epochs_from'], c2['threshold']) == (
        'gliner.training.Trainer', '1.15.0', 'gliner_ft_m300: best_epoch + 1', 0.5)
    assert c2['training_args'] == gb.PACKAGE_ARGS and json.loads(json.dumps(c2)) == c2
    t = c2['training_args']
    assert (t['learning_rate'], t['others_lr'], t['weight_decay'], t['others_weight_decay'],
            t['warmup_ratio'], t['warmup_steps'], t['per_device_train_batch_size'],
            t['gradient_accumulation_steps'], t['max_grad_norm'], t['num_train_epochs'],
            t['max_steps'], t['lr_scheduler_type'], t['optim']) == (
        1e-5, 5e-5, 0.01, 0.01, 0.1, 0, 8, 1, 1.0, 20, -1, 'linear', 'adamw_torch')
    assert (t['adam_beta1'], t['adam_beta2'], t['adam_epsilon']) == (0.9, 0.999, 1e-8)
    assert (t['focal_loss_alpha'], t['focal_loss_gamma'], t['rel_focal_loss_alpha'],
            t['rel_focal_loss_gamma'], t['loss_reduction'], t['masking'], t['negatives'],
            t['loss_type'], t['label_smoothing']) == (-1.0, 0.0, -1.0, 0.0, 'sum', 'none', 1.0,
                                                      'focal', 0.0)
    assert (t['fp16'], t['bf16'], t['full_determinism'], t['eval_strategy'],
            t['save_strategy']) == (False, False, False, 'no', 'no')
    loop = gb.stored_config(gb.parse_args([*FINE_ARGS, '--tag', 'm300', '--seed-offset', '300']),
                            gb.crf_arguments(e2), FINE_LABELS)
    common = [k for k in loop if k in c2]
    assert {k: c2[k] for k in common} == {k: loop[k] for k in common}
    assert set(loop) - set(c2) == {'epochs', 'patience', 'batch_size', 'grad_accumulation',
                                   'lr_encoder', 'lr_others', 'weight_decay', 'warmup_ratio',
                                   'max_grad_norm', 'adam_betas', 'adam_eps', 'loss'}
    for argv, error in (
            ([*FINE_ARGS, '--coarse-names', '--tag', 'm0'], '--coarse-names (E1)'),
            ([*COARSE_ARGS, '--coarse-names', '--zero-shot', '--tag', 'm0'], '--coarse-names (E1)'),
            ([*COARSE_ARGS, '--coarse-names', '--package-trainer', '--tag', 'm0'],
             '--coarse-names (E1)'),
            ([*FINE_ARGS, '--package-trainer', '--zero-shot', '--tag', 'm0'], 'not --zero-shot'),
            ([*FINE_ARGS, '--package-trainer', '--epochs', '5', '--tag', 'm0'],
             'not for --package-trainer'),
            ([*FINE_ARGS, '--package-trainer', '--grad-accumulation', '2', '--tag', 'm0'],
             'not for --package-trainer'),
            ([*FINE_ARGS, '--stored-runs', 'somewhere', '--tag', 'm0'],
             'is for --package-trainer')):
        with pytest.raises(SystemExit):
            gb.parse_args(argv)
        assert error in capsys.readouterr().err


def test_e2_needs_the_pinned_accelerate(monkeypatch):
    """E2 stops before reading any data without accelerate or with another version of it."""
    monkeypatch.setitem(sys.modules, 'accelerate', None)
    with pytest.raises(SystemExit, match='needs accelerate==1.15.0'):
        gb.check_accelerate()
    called = []
    monkeypatch.setattr(nb, 'granularity_data', lambda *a: called.append(a))
    with pytest.raises(SystemExit, match='needs accelerate==1.15.0'):
        gb.run(gb.parse_args([*FINE_ARGS, '--package-trainer', '--tag', 'm0']))
    assert not called
    monkeypatch.setitem(sys.modules, 'accelerate', types.SimpleNamespace(__version__='1.14.0'))
    with pytest.raises(SystemExit, match='accelerate 1.14.0 is installed, not 1.15.0'):
        gb.check_accelerate()
    monkeypatch.setitem(sys.modules, 'accelerate', types.SimpleNamespace(__version__='1.15.0'))
    assert gb.check_accelerate() == '1.15.0'


# --- batch w21x ---------------------------------------------------------------------------------

# the specs of w21s and w21a, their external inputs stubbed, as JSON: its SHA-256 before the code
# of the exploratory checks (commit 1c6f497)
W21_SPECS_SHA256 = '07e98cfce2ffd62bfb7e94368beb2bb2cf1ed5031b4db75ffda3773653a8b2d3'


def test_w21s_and_w21a_are_unchanged(monkeypatch):
    _stub_inputs(monkeypatch)
    specs = {n: kj.batch_spec(n) for n in ('w21s', 'w21a')}
    text = json.dumps(specs, sort_keys=True)
    assert hashlib.sha256(text.encode()).hexdigest() == W21_SPECS_SHA256
    assert kj.GLINER_PINS['pip_no_deps'] == ('gliner==0.2.30',)


def test_the_w21x_batch(monkeypatch):
    """w21x: smoke4's CRF repeat (smokebo), then E1's 10 jobs (w21a's coarse jobs with
    --coarse-names, gliner_ftn_m0 to _m900) and E2's 10 (--package-trainer on fine paper 1,
    gliner_ftp_m0 to _m900), five of each per lane, each its own file, the files that week21x
    reads; w21a's pins with accelerate beside gliner in the --no-deps install."""
    _stub_inputs(monkeypatch)
    smokebh = kj.batch_lanes('w20a')[0][0]
    lane0, lane1 = kj.batch_lanes('w21x')
    assert lane0[0] == smokebh[:-1] + ['smokebo']
    jobs = lane0[1:] + lane1
    parsed = [_gliner_args(j) for j in jobs]
    e1 = [a for a in parsed if a.coarse_names]
    e2 = [a for a in parsed if a.package_trainer]
    assert len(parsed) == len(e1) + len(e2) == 20
    assert sorted(a.seed_offset for a in e1) == sorted(a.seed_offset for a in e2) == list(OFFSETS)
    for a in e1:
        assert (a.granularity, a.units, a.zero_shot, a.package_trainer) == ('coarse', None, False,
                                                                           False)
        assert (a.tag, a.epochs, a.patience, a.grad_accumulation, a.threads, a.max_train) == (
            f'm{a.seed_offset}', 20, 3, 1, 2, 0)
    for a in e2:
        assert (a.granularity, a.units, a.zero_shot, a.coarse_names) == ('fine', [kj.PAPERS[0]],
                                                                         False, False)
        assert (a.tag, a.threads, a.max_train, a.stored_runs) == (f'm{a.seed_offset}', 2, 0,
                                                                  str(gb.STORED_RUNS))
    w21a = [j for lane in kj.batch_lanes('w21a') for j in lane]
    for j in jobs:
        if '--coarse-names' in j:
            assert [x for x in j if x != '--coarse-names'] in w21a
    for lane in (lane0[1:], lane1):
        assert sum('--coarse-names' in j for j in lane) == 5
        assert sum('--package-trainer' in j for j in lane) == 5
    files = [_written(j) for j in lane0 + lane1]
    assert len(set(files)) == 21
    assert {f for f in files if f[2].startswith(sa.WEEK21X_NAMES)} == {
        ('coarse', 'leave_one_paper_out', f'{sa.WEEK21X_NAMES}_m{off}') for off in OFFSETS}
    assert {f for f in files if f[2].startswith(sa.WEEK21X_PACKAGE)} == {
        ('fine', 'leave_one_paper_out', f'{sa.WEEK21X_PACKAGE}_m{off}') for off in OFFSETS}
    assert (gb.NAMES_VARIANT, gb.PACKAGE_VARIANT) == (sa.WEEK21X_NAMES, sa.WEEK21X_PACKAGE)
    assert sa.WEEK21X_E1 == kj.W21X_E1[0] and sa.WEEK21X_E2 == ('fine', kj.W21X_E2[2] + '.conll')
    assert kj.W21X_E2[2] + '.conll' == PAPER1
    assert 'w21x' in kj.BATCHES and kj.batch_name('w21x') == 'w21x'
    spec, w21a_spec = kj.batch_spec('w21x'), kj.batch_spec('w21a')
    assert spec['pip'] == w21a_spec['pip'] and spec['models'] == w21a_spec['models']
    assert spec['pip_no_deps'] == w21a_spec['pip_no_deps'] + [kj.ACCELERATE] == [
        'gliner==0.2.30', f'accelerate=={gb.ACCELERATE_VERSION}']
    assert not {'mulms_files', 'sofc_corpus_sha256', 'topic_corpora', 'week19_names'} & set(spec)
    rj = _run_jobs()
    assert rj.pip_commands(spec, ['w.whl'])['pip_no_deps'] == [
        sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'gliner==0.2.30',
        'accelerate==1.15.0']
    assert all(rj.job_command(j)[1:] == j for j in jobs)


# --- the week21x section of ner_seed_analysis.py, dry-run on fake families ----------------------

# E1: gliner_ftn minus gliner_ft per paper (micro F1); E2: gliner_ftp minus gliner_ft per seed offset
NAMES_SHIFT = {f'{p}.conll': s for p, s in zip(kj.PAPERS, (2.0, 3.5, 1.0, 4.5), strict=True)}
PACKAGE_SHIFT = (0.5, -1.0, 1.5, 0.0, -0.5, 2.0, 1.0, -1.5, 0.5, 1.0)


@pytest.fixture(scope='module')
def dry21x(tmp_path_factory):
    """A results directory with the stored files the checks pair (linked: the CRF head's and
    gliner_ft's on the coarse level, gliner_ft's and crf_m0 on the fine level) and fake families:
    gliner_ftn_m0 to _m900, copies of the coarse gliner_ft files with the class names and micro F1
    shifted by NAMES_SHIFT; gliner_ftp_m0 to _m900, paper 1's run of the fine gliner_ft files with
    micro F1 shifted by PACKAGE_SHIFT, its epochs the stored best epoch + 1, and the trainer and
    accelerate in the config."""
    if not (GPU / f'ner_results_{COARSE}_gliner_ft_m0.json').exists():
        pytest.skip('the stored results are not here')
    results = tmp_path_factory.mktemp('w21x_dry')
    for k, off in enumerate(OFFSETS):
        for name in (f'ner_results_{COARSE}_crf_m{off}.json',
                     f'ner_results_{COARSE}_gliner_ft_m{off}.json',
                     f'ner_results_{FINE}_gliner_ft_m{off}.json'):
            (results / name).symlink_to(GPU / name)
        names = copy.deepcopy(_stored(f'ner_results_{COARSE}_gliner_ft_m{off}.json'))
        names['variant'] = f'gliner_ftn_m{off}'
        names['config']['label_names'] = gb.label_map('coarse', names['labels'],
                                                      coarse_names=True)
        for run in names['runs']:
            run['f1'] = round(run['f1'] + NAMES_SHIFT[run['run_id']], 2)
        _write(results / f'ner_results_{COARSE}_gliner_ftn_m{off}.json', names)
        package = copy.deepcopy(_stored(f'ner_results_{FINE}_gliner_ft_m{off}.json'))
        package['variant'] = f'gliner_ftp_m{off}'
        package['config'].update(trainer=gb.PACKAGE_TRAINER,
                                 accelerate_version=gb.ACCELERATE_VERSION)
        run = next(r for r in package['runs'] if r['run_id'] == PAPER1)
        run.update(f1=round(run['f1'] + PACKAGE_SHIFT[k], 2), epochs=run['best_epoch'] + 1)
        package['runs'] = [run]
        _write(results / f'ner_results_{FINE}_gliner_ftp_m{off}.json', package)
    (results / f'ner_results_{FINE}_crf_m0.json').symlink_to(GPU / f'ner_results_{FINE}_crf_m0.json')
    return results


def _unit_means(prefix, system):
    files = [_stored(f'ner_results_{prefix}_{system}_m{off}.json') for off in OFFSETS]
    return {r['run_id']: statistics.fmean(next(x['f1'] for x in f['runs']
                                               if x['run_id'] == r['run_id']) for f in files)
            for r in files[0]['runs']}


def test_the_week21x_section_on_fake_families(dry21x, tmp_path):
    """E1: gliner_ftn minus the CRF head and minus gliner_ft, each unit the mean of its 10 seeds,
    paired over the 4 papers with Student-t intervals: the differences from gliner_ft are the
    shifts; E2: gliner_ftp minus gliner_ft on paper 1, paired by seed (10 pairs), each side's mean;
    no decision in either; the checks hold; the new files leak into no other section."""
    pytest.importorskip('scipy')
    w = sa.week21x(dry21x)
    assert set(w) == {'checks', 'names', 'package_trainer'}
    checks = w['checks']
    assert (checks['n_files'], checks['n_runs'], checks['splits_equal'],
            checks['environment_equal']) == (20, 10 * 4 + 10, True, True)
    e = w['names']
    units = [r['run_id'] for r in _stored(f'ner_results_{COARSE}_crf_m0.json')['runs']]
    assert (e['setting'], e['n_units'], e['seeds'], list(e['units'])) == ('coarse', 4,
                                                                          list(OFFSETS), units)
    shifts = [NAMES_SHIFT[u] for u in units]
    d = e['minus_gliner_ft']['micro']
    assert d['per_run'] == pytest.approx(shifts, abs=0.011)
    assert d['mean_diff'] == pytest.approx(statistics.mean(shifts), abs=0.011)
    assert d['ci95'] == pytest.approx(metrics.mean_ci(shifts)['ci'], abs=0.02)
    assert e['minus_gliner_ft']['macro']['per_run'] == [0.0] * 4       # micro F1 alone shifted
    crf, ft = _unit_means(COARSE, 'crf'), _unit_means(COARSE, 'gliner_ft')
    assert e['crf']['micro']['per_run'] == [round(crf[u], 2) for u in units]
    assert e['gliner_ft']['micro']['per_run'] == [round(ft[u], 2) for u in units]
    assert e['minus_crf']['micro']['per_run'] == pytest.approx(
        [ft[u] + NAMES_SHIFT[u] - crf[u] for u in units], abs=0.011)
    labels = _stored(f'ner_results_{COARSE}_crf_m0.json')['labels']
    assert e['label_names'] == gb.label_map('coarse', labels, coarse_names=True)
    assert e['label_names_differ'] == []
    assert e['validation']['gliner_ftn'] == e['validation']['gliner_ft']
    p = w['package_trainer']
    assert (p['setting'], p['unit'], p['n_pairs'], p['seeds'], p['seeds_equal'],
            p['epochs_match']) == ('fine', PAPER1, 10, list(OFFSETS), True, True)
    d = p['minus_gliner_ft']['micro']
    assert d['per_run'] == pytest.approx(PACKAGE_SHIFT, abs=0.011)
    assert d['ci95'] == pytest.approx(metrics.mean_ci(PACKAGE_SHIFT)['ci'], abs=0.02)
    loop = [next(r['f1'] for r in _stored(f'ner_results_{FINE}_gliner_ft_m{off}.json')['runs']
                 if r['run_id'] == PAPER1) for off in OFFSETS]
    assert p['gliner_ft']['micro']['per_run'] == loop
    assert p['gliner_ft']['micro']['ci95'] == [round(x, 2) for x in metrics.mean_ci(loop)['ci']]
    assert [s['epochs'] for s in p['per_seed']] == list(STORED_EPOCHS.values())
    assert [s['stored_best_epoch'] + 1 for s in p['per_seed']] == list(STORED_EPOCHS.values())
    assert (p['trainers'], p['accelerate_versions']) == (['gliner.training.Trainer'], ['1.15.0'])
    for entry in (e, p):
        assert not {'passes', 'below', 'reading', 'decided', 'decision', 'dependence'} & set(entry)
    # the new files leak into no other section
    plain = tmp_path / 'plain'
    plain.mkdir()
    for path in dry21x.iterdir():
        if not any(f'_{s}_' in path.name for s in (sa.WEEK21X_NAMES, sa.WEEK21X_PACKAGE)):
            (plain / path.name).symlink_to(path.resolve())
    assert sa.week21x(plain) == {}
    assert sa.week21(dry21x) == sa.week21(plain) != {}
    assert sa.week12_variants(dry21x) == {} and sa.week20(dry21x) == {}


def test_the_week21x_parts_wait_for_their_files_and_report_their_checks(dry21x, tmp_path,
                                                                        capsys):
    """A part waits for every file of its families with its unit; a split that is not the CRF
    head's, another environment, an epoch count that is not the stored one, other seeds, another
    accelerate and other label names are reported, with the numbers."""
    pytest.importorskip('scipy')

    def directory(name, changes):
        out = tmp_path / name
        out.mkdir()
        for path in dry21x.iterdir():
            if path.name in changes:
                if changes[path.name] is not None:
                    _write(out / path.name, changes[path.name])
            else:
                (out / path.name).symlink_to(path.resolve())
        return out

    ftn = f'ner_results_{COARSE}_gliner_ftn_m900.json'
    ftp = f'ner_results_{FINE}_gliner_ftp_m300.json'
    w = sa.week21x(directory('no_ftn', {ftn: None}))
    assert set(w) == {'checks', 'package_trainer'} and w['checks']['n_files'] == 19
    running = json.loads((dry21x / ftn).read_text())
    running['runs'] = running['runs'][:-1]
    assert 'names' not in sa.week21x(directory('running', {ftn: running}))
    no_unit = json.loads((dry21x / ftp).read_text())
    no_unit['runs'][0]['run_id'] = 'fold0'
    assert 'package_trainer' not in sa.week21x(directory('no_unit', {ftp: no_unit}))
    bad = json.loads((dry21x / ftp).read_text())
    run = bad['runs'][0]
    run['test_index'] = run['test_index'][1:]
    run['environment'] = dict(run['environment'], torch='2.12.0')
    run['epochs'] += 1
    run['seed'] = 301
    bad['config']['accelerate_version'] = '1.14.0'
    renamed = json.loads((dry21x / ftn).read_text())
    renamed['config']['label_names'] = {c: c.lower() for c in renamed['labels']}
    w = sa.week21x(directory('bad', {ftp: bad, ftn: renamed}))
    assert w['checks']['splits_differ'] == [f'{ftp} {PAPER1}: test_index']
    assert w['checks']['environment_differs'] == [f'{ftp} {PAPER1}']
    p = w['package_trainer']
    assert (p['epochs_match'], p['seeds_equal'], p['accelerate_versions']) == (False, False,
                                                                               ['1.14.0', '1.15.0'])
    assert 'minus_gliner_ft' in p and w['names']['label_names_differ'] == ['gliner_ftn_m900']
    capsys.readouterr()
    sa.print_week21x(w)
    out = capsys.readouterr().out
    assert f'splits DIFFER: {ftp} {PAPER1}: test_index' in out
    assert f'environment DIFFERS: {ftp} {PAPER1}' in out
    assert 'the label names DIFFER in gliner_ftn_m900' in out
    assert ('accelerate 1.14.0, 1.15.0; the seeds of a pair DIFFER; the epochs DIFFER from the '
            'stored best epochs)') in out


def test_main_prints_the_week21x_section(dry21x, tmp_path, capsys):
    pytest.importorskip('scipy')
    sa.print_week21x(sa.week21x(dry21x))
    out = capsys.readouterr().out
    assert ("week 21x (exploratory), 50 GLiNER runs in 20 files: splits the CRF head's; "
            "environment the stored runs'") in out
    assert 'week 21x, E1 (coarse, the codes as class names; reported, no decision): GLiNER ' \
           'fine-tuned' in out
    assert 'minus GLiNER fine-tuned with the codes lowercased' in out
    assert ("week 21x, E2 (fine, 10.1007_s11340-021-00758-x.conll; reported, no decision): the "
            "package's trainer") in out
    assert "minus this repository's loop" in out and 'over 10 seeds' in out
    assert '(gliner.training.Trainer, accelerate 1.15.0)' in out
    assert 'passes' not in out and 'DIFFER' not in out
    sa.print_week21x({})
    assert capsys.readouterr().out == ''


def test_the_stored_analysis_has_no_week21x_section():
    """Until their files exist, the exploratory checks add nothing: the stored analysis has no
    week21x section, and the stored results give none."""
    stored = GPU / sa.OUT_NAME
    if not stored.exists():
        pytest.skip('the stored results are not here')
    assert 'week21x' not in json.loads(stored.read_text())
    assert sa.week21x(GPU) == {}


# --- E2 on the real model (accelerate 1.15.0, gliner 0.2.30 and the pinned model) ----------------

def _need_e2():
    gliner = pytest.importorskip('gliner')
    if gliner.__version__ != gb.GLINER_VERSION or not _cached_model():
        pytest.skip(f'gliner {gb.GLINER_VERSION} and the pinned model are not here')
    proc = subprocess.run([sys.executable, '-I', '-c',
                           'import accelerate; print(accelerate.__version__)'],
                          capture_output=True, text=True)
    if proc.returncode or proc.stdout.strip() != gb.ACCELERATE_VERSION:
        pytest.skip(f'accelerate {gb.ACCELERATE_VERSION} is not here (E2 needs it)')


PARTS = '''import json, math, sys, tempfile
sys.path.insert(0, {src!r})
import torch
from gliner.training import Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup
import gliner_baseline as gb
sentences = json.loads({toy!r})
labels = ['DEVICE', 'EXPERIMENT', 'MATERIAL', 'VALUE']
names = gb.label_map('sofc', labels)
prompts = [names[c] for c in labels]
model = gb.load_model('cpu')
plain = [{{'tokenized_text': w, 'ner': gb.gold_spans(t, names)}} for w, t in sentences]
labelled = [dict(e, ner_labels=list(prompts)) for e in plain]
collator = model._create_data_collator()
loop_batch = collator(plain, entity_types=[list(prompts) for _ in plain])
package_batch = collator(labelled)
out = {{'batch': set(loop_batch) == set(package_batch) and all(
    torch.equal(loop_batch[k], package_batch[k]) if isinstance(loop_batch[k], torch.Tensor)
    else loop_batch[k] == package_batch[k] for k in loop_batch)}}
model.eval()
with torch.no_grad():
    loop_loss = model.model(**gb.forward_inputs(model, loop_batch, 'cpu'), **gb.LOSS).loss
with tempfile.TemporaryDirectory() as directory:
    args = TrainingArguments(output_dir=directory, seed=0, use_cpu=True, **gb.PACKAGE_ARGS)
    trainer = Trainer(model=model, args=args, train_dataset=labelled, data_collator=collator,
                      processing_class=model.data_processor.transformer_tokenizer)
    model.eval()
    with torch.no_grad():
        package_loss = trainer.compute_loss(model, trainer._prepare_inputs(package_batch))
    out['loss'] = bool(torch.equal(loop_loss, package_loss))
    optimizer = trainer.create_optimizer()
    ours = torch.optim.AdamW(gb.parameter_groups(model), lr=gb.LR_ENCODER, betas=gb.ADAM_BETAS,
                             eps=gb.ADAM_EPS)
    name_of = {{id(p): n for n, p in model.named_parameters()}}
    def groups(param_groups):
        return sorted((g['lr'], g['weight_decay'], tuple(sorted(name_of[id(p)] for p in g['params'])))
                      for g in param_groups if g['params'])
    out['groups'] = groups(optimizer.param_groups) == groups(ours.param_groups)
    out['optimizer'] = (type(optimizer).__name__, optimizer.defaults['lr'],
                        list(optimizer.defaults['betas']), optimizer.defaults['eps'])
    steps = gb.EPOCHS * math.ceil(370 / gb.BATCH_SIZE)
    scheduler = trainer.create_scheduler(num_training_steps=steps, optimizer=optimizer)
    reference = get_linear_schedule_with_warmup(ours, math.ceil(gb.WARMUP_RATIO * steps), steps)
    same = True
    for _ in range(steps):
        same &= sorted(g['lr'] for g in optimizer.param_groups) == sorted(
            g['lr'] for g in ours.param_groups)
        scheduler.step()
        reference.step()
    out.update(schedule=same, steps=steps, warmup=args.get_warmup_steps(steps))
print(json.dumps(out))
'''


def test_the_package_trainers_parts_are_the_loops():
    """E2's trainer differs from gliner_ft's loop in its order of the batches and its random state
    only: on the real model in eval mode (no dropout), the collated batch (the labels as
    ner_labels against entity_types), the loss of a batch, the optimizer's parameter groups and
    the learning rate at every step of paper 1's schedule (940 steps, 94 of warm-up) are the
    loop's."""
    _need_e2()
    toy = [(['the', 'anode', 'was', 'nickel', 'at', '800', 'K'],
            ['O', 'B-DEVICE', 'O', 'B-MATERIAL', 'O', 'B-VALUE', 'I-VALUE']),
           (['we', 'measured', 'the', 'cell'], ['O', 'B-EXPERIMENT', 'O', 'B-DEVICE']),
           (['zirconia', 'layers', 'were', 'tested'], ['B-MATERIAL', 'O', 'O', 'B-EXPERIMENT'])]
    code = PARTS.format(src=str(REPO / 'src' / 'graph'), toy=json.dumps(toy * 3))
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    proc = subprocess.run([sys.executable, '-I', '-c', code], env=env, capture_output=True,
                          text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out == {'batch': True, 'loss': True, 'groups': True,
                   'optimizer': ['AdamW', 1e-5, [0.9, 0.999], 1e-8], 'schedule': True,
                   'steps': 940, 'warmup': 94}


def test_a_tiny_cpu_run_of_e2(tmp_path):
    """On the toy corpus, offline: gliner_ft (this repository's loop, 3 epochs at most) writes the
    stored run; E2 trains its best epoch + 1 by the package's trainer, on the same split, with the
    learning rate of gliner_ft's 20-epoch schedule at each epoch; its config records the trainer
    and accelerate; it resumes."""
    _need_e2()
    _drive(tmp_path, '--tag', 'm0', '--epochs', '3')
    loop, = json.loads((tmp_path / 'ner' / 'ner_results_sofc_document_folds_gliner_ft_m0.json')
                       .read_text())['runs']
    flags = ('--package-trainer', '--stored-runs', str(tmp_path / 'ner'), '--tag', 'm0')
    _drive(tmp_path, *flags)
    path = tmp_path / 'ner' / 'ner_results_sofc_document_folds_gliner_ftp_m0.json'
    res = json.loads(path.read_text())
    config = res['config']
    assert (res['variant'], config['trainer'], config['accelerate_version'],
            config['epochs_from'], config['training_args']) == (
        'gliner_ftp_m0', 'gliner.training.Trainer', '1.15.0', 'gliner_ft_m0: best_epoch + 1',
        gb.PACKAGE_ARGS)
    run, = res['runs']
    assert list(run) == ['f1', 'precision', 'recall', 'f1_strict', 'per_class', 'seed',
                         'best_epoch', 'epochs', 'val_f1', 'n_train', 'n_val', 'n_test', 'history',
                         'test_index', 'test_gold', 'test_pred', 'run_id', 'runtime_sec']
    assert (run['run_id'], run['seed'], run['epochs'], run['best_epoch']) == (
        'fold0', 0, loop['best_epoch'] + 1, loop['best_epoch'])
    for key in ('test_index', 'test_gold', 'n_train', 'n_val', 'n_test'):
        assert run[key] == loop[key]
    # one step per epoch (5 training sentences): the schedule spans 20 steps, 2 of warm-up
    steps, warmup = gb.EPOCHS, 2
    expected = [gb.LR_OTHERS * (e / warmup if e < warmup else (steps - e) / (steps - warmup))
                for e in range(run['epochs'])]
    assert [h['epoch'] for h in run['history']] == list(range(run['epochs']))
    assert [h['learning_rate'] for h in run['history']] == pytest.approx(expected)
    assert 'already done' in _drive(tmp_path, *flags)
    assert json.loads(path.read_text())['runs'] == res['runs']
