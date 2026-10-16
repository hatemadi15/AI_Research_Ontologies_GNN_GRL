"""
kaggle_jobs.py - NER runs on Kaggle GPUs (ROADMAP §4, week-4 protocol)

Drives the Kaggle command-line tool (`kaggle` on the PATH, or the path in KAGGLE_CLI),
which signs in with the token in KAGGLE_API_TOKEN; the account name is KAGGLE_USERNAME
or --user. Every run of week 4 trains on Kaggle:

  bundle   build the bundle dataset in data/processed/kaggle/bundle/: code.tar.gz (git
           archive of a frozen commit), fatigue_sentences.txt (its SHA-256 checked against
           results/fatigue_corpus.json), the GPU results so far (results/gpu/ner_results_*
           as gpu__ner_results_*.json, so that finished runs are skipped), wheels of the
           packages that PyPI has as source only (WHEELS, built here: Kaggle's Python
           cannot build them) and BUNDLE.json (the commit, the corpus SHA-256, the results
           files); then create the private dataset, or add a version, and wait until
           Kaggle has processed it
  push     write a batch's job spec (batch_spec) into src/kaggle/run_jobs.py and push it
           as a private GPU kernel: NvidiaTeslaT4 (two T4 GPUs, one lane each), internet on
           (pinned packages and model snapshot), the bundle attached, and the Kaggle image
           pinned to the kernel's first version
  status   the kernel's status
  pull     download the kernel's outputs to data/processed/kaggle/<kernel>/<time>/ and
           print the manifest: setup errors, and per job its status and minutes
  collect  copy the results files of a pulled output to results/gpu/ (smoke-test files
           excepted). Every run must be a GPU run with one environment (GPU, CUDA, torch,
           transformers) shared by every run in results/gpu/, and a file there is only
           replaced by one that holds all its runs unchanged
  quota    the weekly GPU hours used and left

Batches (`push --batch NAME`), each a list of ner_baseline.py jobs on two lanes; a job
writes its own results files, so the lanes never write the same file:
  smoke1  the GPU check: one CRF run three times (GPU 0, GPU 1, GPU 1 again), a
          multi-task run and a label-semantics run
  smoke2  the smoke tests of the new options (label semantics on the multi-task model,
          self-training with 3 teachers), and the CRF run once more
  smoke3  the label-semantics smoke tests again once rdflib is installed (multi-task, and
          fine-grained alone)
  item1   the CRF reference crf_m0/m100/m200 and multi-task crf_mt/_m100/_m200
  item2   label semantics crf_ls_m* and label semantics on the multi-task model crf_mt_ls_m*
  item12  item1 then item2 in one session (a Kaggle GPU session takes 10-15 minutes to
          start): each lane runs its item1 jobs before its item2 jobs
  item3   week-3 self-training crf_st_m* and self-training with 3 teachers crf_st3_m*,
          held-out papers; the 3-teacher runs one job per paper (--papers)
  item3r  self-training with 3 teachers on the random splits (if the quota allows); a run
          saves its files after every split
Every job has --crf, --save-predictions and --threads 2; the families have 3 seeds per
split (seed offsets 0, 100 and 200) on the 5 random splits and the 4 held-out papers.
Week 5 makes 10 seeds per split (offsets 0 to 900):
  smoke4  a tiny 3-teacher run with 4 students (--students), and the CRF run of split 0
          (fine-grained, seed 0) as smokee, which must equal the stored crf_m0 run
  w5a     self-training with 3 teachers and 10 students: the teachers and the 7 students
          that the week-4 files lack, per split; one lane per granularity, the held-out
          papers first
  w5b     the CRF run of smoke4 again (smokef), then crf_m*, crf_ls_m* and crf_mt_m* at
          seed offsets 300 to 900
  w5c     the published split (week-5 amendment): the CRF run of smoke4 again (smokeg),
          then plain MatSciBERT and the CRF head, model seeds 0 to 900, a lane per
          granularity
  smoke5  the CRF run of smoke4 again (smokel), then the week-6 options, tiny: DeBERTa-v3-large
          twice (with its pins, BATCH_EXTRAS), document context, multi-task self-training and
          SOFC-Exp
  smoke6  the CRF run of smoke4 again (smokem), then one full-size run of DeBERTa-v3-large and
          of document context on the published split, and of SOFC-Exp on its official split:
          splits that no week-6 decision uses
Week 6 (methods for unseen papers, and SOFC-Exp; each session first runs the CRF run of
smoke4 again):
  w6a     the screen of DeBERTa-v3-large (crf_deberta_m*) and document context (crf_ctx_m*):
          seed offsets 0, 100 and 200 on the held-out papers, both granularities
  w6b     the screen of multi-task self-training (crf_mt_st3_m*, 3 students), one job per
          held-out paper in one lane (they extend the same files); SOFC-Exp's official
          split with plain MatSciBERT and the CRF head, model seeds 0 to 900
  w6c:M   stage 2 for the methods M (comma-separated): deberta and ctx add seed offsets 300
          to 900 on the held-out papers; mtst trains 10 students per paper (one lane);
          sofc-deberta and sofc-ctx run the method on SOFC-Exp's official split, model seeds
          0 to 900. E.g. w6c:deberta,mtst. sofc-ctx runs from the later commit of the
          week-6 amendment, so its session also repeats session b's SOFC-Exp CRF run of
          seed 0 (smoket), the reference of its runs. sofc-mtst (alone in its batch):
          multi-task self-training on SOFC-Exp's two levels with the SOFC corpus
          (SOFC_CORPUS, in the bundle as sofc_sentences.txt), its official split; each lane
          trains the 3 teachers and 5 of the 10 students (--student-offsets), after the
          repeats smokeaa (the CRF run of smoke4) and smokeab (session b's SOFC-Exp CRF run)
Week 7, part 2a (recall calibration; the session first repeats the CRF run of smoke4,
smokeu):
  w7a     the CRF head with the bias grid (--entity-bias BIAS_GRID): the inner folds
          (--leave-two-papers-out, seed offsets 0, 100 and 200; counts only), the held-out
          papers again (seed offsets 0 to 900, with predictions) and SOFC-Exp's official
          split (model seeds 0 to 900), both granularities, balanced over the lanes
Week 7, part 2d (SOFC-Exp's two levels):
  w7b     the CRF run of smoke4 again (smokev), session b's SOFC-Exp CRF run of seed 0
          (smokew) and the MaterioMiner multi-task run of split 0 (smokex), each to be
          equal to its stored run; then SOFC-Exp's official split, model seeds 0 to 900:
          plain MatSciBERT and the CRF head on the slot types (sofc_slot), and multi-task
          training on both levels (sofc_both, a file per level)
Week 7, part 2b (mention replacement, --augment-mentions 3; each session first repeats the
CRF run of smoke4):
  w7c     the screen: crf_mr_m0 to _m200 on the held-out papers, both granularities (smokey)
  w7d     stage 2: seed offsets 300 to 900 on the held-out papers, and SOFC-Exp's official
          split with model seeds 0 to 900 (S2), balanced over the lanes (smokez)
Week 7, part 2e (SOFC-Exp's test tokens):
  w7e     the CRF run of smoke4 again (smokeac) and session b's SOFC-Exp CRF run of seed 0
          (smokead); then the CRF head (lane 0) and plain MatSciBERT (lane 1) on the official
          split, model seeds 0 to 900, scored on the test articles cut into words as the
          training articles are (--retokenized-test, variants crf_cdetest and cdetest)
Week 8, part 1 (SOFC-Exp beyond its official split):
  w8a     the CRF run of smoke4 again (smokeae) and session b's SOFC-Exp CRF run of seed 0
          (smokeaf); then the CRF head on SOFC-Exp, model seed offsets 0 to 900 (crf_m<N>),
          on its 5 random sentence splits (lane 0) and its 5 article folds (lane 1,
          --document-folds 5)
Week 8, part 2 (more papers or more sentences?):
  w8b     the CRF run of smoke4 again (smokeag); then the CRF head on the held-out papers with
          --two-paper-size (crf_p3n2_m<N>), seed offsets 0, 100 and 200, one lane per
          granularity

Usage:
  python src/graph/kaggle_jobs.py quota
  python src/graph/kaggle_jobs.py bundle --commit HEAD
  python src/graph/kaggle_jobs.py push --batch smoke1
  python src/graph/kaggle_jobs.py status
  python src/graph/kaggle_jobs.py pull
  python src/graph/kaggle_jobs.py collect data/processed/kaggle/ner-gpu-jobs/<time>
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_JOBS = ROOT / 'src' / 'kaggle' / 'run_jobs.py'
WORK_DIR = ROOT / 'data' / 'processed' / 'kaggle'
GPU_RESULTS = ROOT / 'results' / 'gpu'
CORPUS = 'data/processed/corpus/fatigue_sentences.txt'
CORPUS_REPORT = ROOT / 'results' / 'fatigue_corpus.json'
# the SOFC corpus (week-6 amendment: multi-task self-training on SOFC-Exp, build_sofc_corpus.py)
SOFC_CORPUS = 'data/processed/corpus/sofc_sentences.txt'
SOFC_CORPUS_REPORT = ROOT / 'results' / 'sofc_corpus.json'

DATASET = 'ner-gpu-bundle'
KERNEL = 'ner-gpu-jobs'
MACHINE = 'NvidiaTeslaT4'           # Kaggle's "GPU T4 x2"
# the Kaggle image of the first job (smoke1), pinned for every kernel: one environment
DOCKER_IMAGE = ('gcr.io/kaggle-gpu-images/python@sha256:'
                '2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f')
GPU = 'Tesla T4'
LANES = 2
# the versions of the CPU runs; torch is the one of the Kaggle image
PIP = ('transformers==5.18.0', 'tokenizers==0.23.2', 'huggingface_hub==1.33.0',
       'safetensors==0.8.0', 'rdflib==7.6.0')     # rdflib: the MMO definitions (label semantics)
# source-only on PyPI: built here into pure-Python wheels that go into the bundle
WHEELS = ('seqeval==1.2.2',)
WHEEL_FILES = ('seqeval-1.2.2-py3-none-any.whl',)
MODELS = {'m3rg-iitd/matscibert': 'ced9d8f5f208712c4a90f98a246fe32155b29995'}
# a batch's extra pins (week-6 smoke tests): DeBERTa-v3-large, and sentencepiece, which its
# tokenizer needs (its snapshot has no tokenizer.json)
DEBERTA = 'microsoft/deberta-v3-large'
DEBERTA_PINS = {'pip': ('sentencepiece==0.2.2',),
                'models': {DEBERTA: '64a8c8eab3e352a784c658aef62be1662607476f'}}
BATCH_EXTRAS = {'smoke5': DEBERTA_PINS, 'smoke6': DEBERTA_PINS, 'w6a': DEBERTA_PINS}
THREADS = 2
DEADLINE_HOURS = 10                 # no job starts later; a Kaggle session lasts 12 hours
RESULTS_PREFIX = 'gpu__'            # as in run_jobs.py
PLACEHOLDER = 'JOBS = None   # the job spec; kaggle_jobs.py push replaces this line'

PROTOCOL_FLAGS = {'random_split_vary': ('--vary-split',),
                  'leave_one_paper_out': ('--leave-one-paper-out',)}
RUNS = {'random_split_vary': 5, 'leave_one_paper_out': 4}
GRANULARITIES = ('fine', 'coarse')
OFFSETS = (0, 100, 200)
OFFSETS_W5 = tuple(range(300, 1000, 100))   # week 5: the seeds added to make 10 per split
OFFSETS_PUBLISHED = tuple(range(0, 1000, 100))   # the model seeds on the published split
PAPERS =('10.1007_s11340-021-00758-x', '10.1038_ncomms7164', '10.1038_s41467-020-19071-7',
          '10.3390_met10111458')      # the held-out papers (the MaterioMiner files)
# relative GPU time of one run of each kind (a CRF run is 1), for balancing the lanes
COST = {'crf': 1.0, 'mt': 1.7, 'ls': 1.1, 'mt_ls': 1.8, 'st': 2.5, 'st3': 22.0,
        'deberta': 4.4, 'ctx': 2.7, 'sofc': 2.7,   # sofc: 3.3 min per CRF run (session b, w6b)
        'mr': 3.8}           # mention replacement, 3 copies: about 4 times the training sentences
# week 7, part 2a: the biases that --entity-bias decodes besides 0 (ROADMAP §4)
BIAS_GRID = ('-1', '-0.5', '0.25', '0.5', '0.75', '1', '1.25', '1.5', '2', '2.5', '3')
# week 6: the options of each method on the held-out papers, and the stage-2 parts
METHOD_FLAGS = {'deberta': ('--model', DEBERTA), 'ctx': ('--context', '64'),
                'mr': ('--augment-mentions', '3')}
STAGE2 = ('deberta', 'ctx', 'mtst', 'sofc-deberta', 'sofc-ctx', 'sofc-mtst')
# the two lanes of sofc-mtst: each trains the 3 teachers and half of the 10 students
SOFC_MTST_LANES = (OFFSETS_PUBLISHED[:5], OFFSETS_PUBLISHED[5:])
# week 8, part 1: SOFC-Exp's random sentence splits and article folds (not in PROTOCOL_FLAGS,
# whose protocols the MaterioMiner families run)
SOFC_SPLITS = {'random_split_vary': ('--vary-split',), 'document_folds': ('--document-folds', '5')}


# ---------------------------------------------------------------------------
# Jobs and batches
# ---------------------------------------------------------------------------
def job(granularity, protocol, *flags, offset=None, tag=None):
    """ner_baseline.py arguments of one job: --crf, the options, the seed tag and offset."""
    args = ['--granularity', granularity, *PROTOCOL_FLAGS[protocol], '--crf', *flags]
    if tag is not None:
        args += ['--tag', tag]
    if offset:
        args += ['--seed-offset', str(offset)]
    return args + ['--save-predictions', '--threads', str(THREADS)]


def family(kind, flags, protocols=tuple(PROTOCOL_FLAGS), granularities=GRANULARITIES,
           untagged_zero=False, offsets=OFFSETS):
    """(cost, job) per protocol, granularity and seed offset of a family of seeds.

    Seed tags are m<offset> (m0, m100 and m200 by default; week 5 adds m300 to m900);
    with untagged_zero the offset-0 member has no tag (multi-task: crf_mt, crf_mt_m100,
    crf_mt_m200, as on the CPU).
    """
    out = []
    for protocol in protocols:
        for granularity in granularities:
            for offset in offsets:
                tag = None if untagged_zero and offset == 0 else f'm{offset}'
                out.append((COST[kind] * RUNS[protocol],
                            job(granularity, protocol, *flags, offset=offset, tag=tag)))
    return out


def method_jobs(method, offsets):
    """(cost, job) of a method's seeds on the held-out papers, both granularities: week 6's
    DeBERTa-v3-large (crf_deberta_m<N>) or document context (crf_ctx_m<N>), week 7's mention
    replacement (crf_mr_m<N>)."""
    return [(COST[method] * RUNS['leave_one_paper_out'],
             job(g, 'leave_one_paper_out', *METHOD_FLAGS[method], offset=off,
                 tag=f'deberta_m{off}' if method == 'deberta' else f'm{off}'))
            for g in GRANULARITIES for off in offsets]


def mtst_jobs(students):
    """Multi-task self-training on the held-out papers (crf_mt_st3_m*, both granularities),
    one job per paper: 3 multi-task teachers tag the whole corpus, `students` students."""
    return [['--leave-one-paper-out', '--papers', paper, '--crf', '--multi-task',
             '--self-train', CORPUS, '--teachers', '3', '--students', str(students),
             '--self-train-sample', '0', '--save-predictions', '--threads', str(THREADS)]
            for paper in PAPERS]


def published_job(granularity, *flags):
    """One job on the published split (SOFC-Exp: its official split), model seeds 0 to 900."""
    return ['--granularity', granularity, '--published-split', '--seeds',
            *(str(s) for s in OFFSETS_PUBLISHED), *flags, '--save-predictions', '--threads',
            str(THREADS)]


def sofc_mtst_job(student_offsets):
    """Multi-task self-training on SOFC-Exp's two levels (crf_mt_st3_m*, a slot-level and an
    entity-level file per student): its official split, model seed 0 + offset; 3 teachers
    tag the whole SOFC corpus, then the students of `student_offsets` (of 10) train."""
    return ['--granularity', 'sofc_both', '--published-split', '--seeds', '0', '--crf',
            '--multi-task', '--self-train', SOFC_CORPUS, '--teachers', '3', '--students', '10',
            '--student-offsets', *(str(o) for o in student_offsets), '--self-train-sample', '0',
            '--save-predictions', '--threads', str(THREADS)]


def sofc_split_job(protocol, offset):
    """The CRF head on SOFC-Exp's random splits or article folds (week 8, part 1), with the
    model seeds of offset `offset` (crf_m<offset>)."""
    return ['--granularity', 'sofc', *SOFC_SPLITS[protocol], '--crf', '--tag', f'm{offset}',
            *(('--seed-offset', str(offset)) if offset else ()), '--save-predictions',
            '--threads', str(THREADS)]


def stage2_methods(name):
    """The methods of a stage-2 batch name 'w6c:M1,M2,...' (ValueError if one is unknown,
    or if sofc-mtst, which takes both lanes, comes with another)."""
    methods = name[len('w6c:'):].split(',')
    unknown = [m for m in methods if m not in STAGE2]
    if not methods or unknown or len(set(methods)) < len(methods):
        raise ValueError(f'unknown or repeated stage-2 methods {name!r}; known: {STAGE2}')
    if 'sofc-mtst' in methods and len(methods) > 1:
        raise ValueError('sofc-mtst takes both lanes of its session: w6c:sofc-mtst alone')
    return methods


def balance(costed, lanes=LANES, loads=None):
    """Jobs over `lanes`: longest first, each to the least loaded lane (ties: lowest);
    `loads` gives the lanes' starting loads."""
    loads = [0.0] * lanes if loads is None else list(loads)
    out = [[] for _ in range(lanes)]
    for cost, args in sorted(costed, key=lambda c: -c[0]):
        lane = loads.index(min(loads))
        out[lane].append(args)
        loads[lane] += cost
    return out


def batch_lanes(name):
    """The jobs of a batch, per lane (see the module docstring)."""
    st = ('--self-train', CORPUS)
    st3 = (*st, '--teachers', '3', '--self-train-sample', '0')
    lopo = ('leave_one_paper_out',)
    if name == 'smoke1':
        crf = ['--granularity', 'fine', '--vary-split', '--seeds', '0', '--crf',
               '--save-predictions', '--threads', str(THREADS)]
        return [[crf + ['--tag', 'smokea'],
                 ['--vary-split', '--seeds', '0', '--crf', '--multi-task', '--tag', 'smoke',
                  '--save-predictions', '--threads', str(THREADS)]],
                [crf + ['--tag', 'smokeb'],
                 ['--granularity', 'coarse', '--leave-one-paper-out', '--papers',
                  '10.3390_met10111458', '--crf', '--label-semantics', '--tag', 'smoke',
                  '--save-predictions', '--threads', str(THREADS)],
                 crf + ['--tag', 'smokec']]]
    if name == 'smoke2':
        tiny = ('--max-train', '32', '--epochs', '2')
        return [[['--vary-split', '--seeds', '0', '--crf', '--multi-task', '--label-semantics',
                  *tiny, '--save-predictions', '--threads', str(THREADS)],
                 ['--granularity', 'fine', '--vary-split', '--seeds', '0', '--crf',
                  '--tag', 'smoked', '--save-predictions', '--threads', str(THREADS)]],
                [['--granularity', 'coarse', '--leave-one-paper-out', '--papers',
                  '10.3390_met10111458', '--crf', *st, '--teachers', '3',
                  '--self-train-sample', '200', *tiny, '--save-predictions', '--threads',
                  str(THREADS)]]]
    if name == 'smoke3':
        tiny = ('--max-train', '32', '--epochs', '2')
        return [[['--vary-split', '--seeds', '0', '--crf', '--multi-task', '--label-semantics',
                  *tiny, '--save-predictions', '--threads', str(THREADS)]],
                [['--granularity', 'fine', '--vary-split', '--seeds', '0', '--crf',
                  '--label-semantics', *tiny, '--save-predictions', '--threads', str(THREADS)]]]
    if name == 'item1':
        return balance(family('crf', ()) +
                       family('mt', ('--multi-task',), granularities=('both',),
                              untagged_zero=True))
    if name == 'item2':
        return balance(family('ls', ('--label-semantics',)) +
                       family('mt_ls', ('--multi-task', '--label-semantics'),
                              granularities=('both',)))
    if name == 'item12':
        return [first + second for first, second in zip(batch_lanes('item1'),
                                                         batch_lanes('item2'))]
    if name == 'item3':
        # one lane per granularity, so that the jobs that write a file share a lane; the
        # 3-teacher runs go paper by paper (each job adds one held-out paper to its files)
        return [[args for _, args in family('st', st, protocols=lopo, granularities=(g,))] +
                [job(g, 'leave_one_paper_out', '--papers', paper, *st3) for paper in PAPERS]
                for g in GRANULARITIES]
    if name == 'item3r':
        # one job per granularity: --seeds K jobs would leave the last K as the stored seeds
        return [[job(g, 'random_split_vary', *st3)] for g in GRANULARITIES]
    # week 5 (ROADMAP §4): 10 seeds per split
    repeat = ['--granularity', 'fine', '--vary-split', '--seeds', '0', '--crf',
              '--save-predictions', '--threads', str(THREADS)]
    if name == 'smoke4':
        tiny = ('--max-train', '32', '--epochs', '2')
        return [[repeat + ['--tag', 'smokee']],
                [['--granularity', 'coarse', '--leave-one-paper-out', '--papers',
                  '10.3390_met10111458', '--crf', *st, '--teachers', '3', '--students', '4',
                  '--self-train-sample', '200', *tiny, '--save-predictions', '--threads',
                  str(THREADS)]]]
    if name == 'w5a':
        # one lane per granularity (its jobs write its files); the held-out papers first
        return [[job(g, protocol, *st3, '--students', '10')
                 for protocol in ('leave_one_paper_out', 'random_split_vary')]
                for g in GRANULARITIES]
    if name == 'w5b':
        lanes = balance(family('crf', (), offsets=OFFSETS_W5) +
                        family('ls', ('--label-semantics',), offsets=OFFSETS_W5) +
                        family('mt', ('--multi-task',), granularities=('both',),
                               offsets=OFFSETS_W5))
        return [[repeat + ['--tag', 'smokef']] + lanes[0], lanes[1]]
    if name == 'smoke5':
        # the week-6 options (prepared in week 5): DeBERTa-v3-large twice (memory and
        # repeatability), document context, multi-task self-training and SOFC-Exp, all tiny;
        # first the week-4 run again, with the new code and the extra pins
        tiny = ('--max-train', '32', '--epochs', '2', '--save-predictions', '--threads',
                str(THREADS))
        deberta = ['--granularity', 'fine', '--vary-split', '--seeds', '0', '--crf', '--model',
                   DEBERTA, *tiny]
        paper = ('--leave-one-paper-out', '--papers', '10.3390_met10111458')
        return [[repeat + ['--tag', 'smokel'], deberta + ['--tag', 'smokeh'],
                 deberta + ['--tag', 'smokei']],
                [['--granularity', 'coarse', *paper, '--crf', '--context', '64', '--tag',
                  'smokej', *tiny],
                 [*paper, '--crf', '--multi-task', '--self-train', CORPUS, '--teachers', '3',
                  '--students', '4', '--self-train-sample', '200', *tiny],
                 ['--granularity', 'sofc', '--leave-one-paper-out', '--papers', 'PMC3564701',
                  '--crf', '--tag', 'smokek', *tiny]]]
    if name == 'smoke6':
        # the tiny runs of smoke5 learn nothing (F1 0 for every option), so one run of each
        # option at full size, seed 0, on splits that no week-6 decision uses
        published = ('--published-split', '--seeds', '0', '--crf', '--save-predictions',
                     '--threads', str(THREADS))
        return [[repeat + ['--tag', 'smokem'],
                 ['--granularity', 'fine', *published, '--model', DEBERTA, '--tag', 'smoken']],
                [['--granularity', 'fine', *published, '--context', '64', '--tag', 'smokeo'],
                 ['--granularity', 'sofc', *published, '--tag', 'smokep']]]
    if name == 'w5c':
        # the published split (week-5 amendment): plain MatSciBERT and the CRF head, 10 seeds
        return [[repeat + ['--tag', 'smokeg'], published_job('fine'),
                 published_job('fine', '--crf')],
                [published_job('coarse'), published_job('coarse', '--crf')]]
    # week 6 (ROADMAP §4): the screen, then stage 2 for the methods that pass it
    if name == 'w6a':
        lanes = balance(method_jobs('deberta', OFFSETS) + method_jobs('ctx', OFFSETS))
        return [[repeat + ['--tag', 'smokeq']] + lanes[0], lanes[1]]
    if name == 'w6b':
        return [[repeat + ['--tag', 'smoker'], *mtst_jobs(3)],
                [published_job('sofc'), published_job('sofc', '--crf')]]
    if name == 'w6c:sofc-mtst':
        # the week-6 amendment's S2 for multi-task self-training: from its own commit, so the
        # session repeats session b's SOFC-Exp CRF run of seed 0 (smokeab); its teachers must
        # equal the multi-task runs of seeds 0, 100 and 200 of part 2d (w7b)
        sofc_repeat = ['--granularity', 'sofc', '--published-split', '--seeds', '0', '--crf',
                       '--tag', 'smokeab', '--save-predictions', '--threads', str(THREADS)]
        return [[repeat + ['--tag', 'smokeaa'], sofc_repeat, sofc_mtst_job(SOFC_MTST_LANES[0])],
                [sofc_mtst_job(SOFC_MTST_LANES[1])]]
    if name.startswith('w6c:'):
        # jobs that extend the same files share lane 1 (mtst); the others are balanced
        methods = stage2_methods(name)
        fixed = mtst_jobs(10) if 'mtst' in methods else []
        costed = [c for m in ('deberta', 'ctx') if m in methods
                  for c in method_jobs(m, OFFSETS_W5)]
        costed += [(10 * COST['deberta'] * 3, published_job('sofc', '--crf', '--model', DEBERTA,
                                                            '--tag', 'deberta'))
                   for m in methods if m == 'sofc-deberta']
        costed += [(10 * COST['ctx'] * 3, published_job('sofc', '--crf', '--context', '64'))
                   for m in methods if m == 'sofc-ctx']
        lanes = balance(costed, loads=[1.0, 70.0 * len(fixed)])
        first = [repeat + ['--tag', 'smokes']]
        if 'sofc-ctx' in methods:
            # the amendment's later commit must give session b's SOFC-Exp CRF run again
            first.append(['--granularity', 'sofc', '--published-split', '--seeds', '0', '--crf',
                          '--tag', 'smoket', '--save-predictions', '--threads', str(THREADS)])
        return [first + lanes[0], fixed + lanes[1]]
    if name == 'w7b':
        # week 7, part 2d: SOFC-Exp's slot level and multi-task training on its two levels;
        # the repeats show that this commit leaves the stored SOFC-Exp CRF runs and the
        # MaterioMiner multi-task runs as they were
        sofc_repeat = ['--granularity', 'sofc', '--published-split', '--seeds', '0', '--crf',
                       '--tag', 'smokew', '--save-predictions', '--threads', str(THREADS)]
        mt_repeat = ['--vary-split', '--seeds', '0', '--crf', '--multi-task', '--tag',
                     'smokex', '--save-predictions', '--threads', str(THREADS)]
        costed = [(COST['sofc'] * len(OFFSETS_PUBLISHED), published_job('sofc_slot')),
                  (COST['sofc'] * len(OFFSETS_PUBLISHED), published_job('sofc_slot', '--crf')),
                  (COST['sofc'] * COST['mt'] * len(OFFSETS_PUBLISHED),
                   published_job('sofc_both', '--crf', '--multi-task'))]
        lanes = balance(costed, loads=[COST['crf'] + COST['sofc'], COST['mt']])
        return [[repeat + ['--tag', 'smokev'], sofc_repeat] + lanes[0], [mt_repeat] + lanes[1]]
    if name in ('w7c', 'w7d'):
        # week 7, part 2b: mention replacement, the screen (seeds 0-200) and stage 2
        costed = method_jobs('mr', OFFSETS if name == 'w7c' else OFFSETS_W5)
        if name == 'w7d':
            costed.append((COST['sofc'] * COST['mr'] * len(OFFSETS_PUBLISHED),
                           published_job('sofc', '--crf', *METHOD_FLAGS['mr'])))
        lanes = balance(costed, loads=[COST['crf'], 0.0])
        smoke = 'smokey' if name == 'w7c' else 'smokez'
        return [[repeat + ['--tag', smoke]] + lanes[0], lanes[1]]
    if name == 'w7e':
        # week 7, part 2e: SOFC-Exp's test articles cut as its training articles are; the
        # runs train as session b's (their training must equal it, seed by seed)
        sofc_repeat = ['--granularity', 'sofc', '--published-split', '--seeds', '0', '--crf',
                       '--tag', 'smokead', '--save-predictions', '--threads', str(THREADS)]
        retokenized = ('--retokenized-test', '--tag', 'cdetest')
        return [[repeat + ['--tag', 'smokeac'], sofc_repeat,
                 published_job('sofc', '--crf', *retokenized)],
                [published_job('sofc', *retokenized)]]
    if name == 'w8a':
        # week 8, part 1: SOFC-Exp's random splits (lane 0) and article folds (lane 1), the
        # CRF head with 10 model seeds; first the repeats of the CRF run of smoke4 and of
        # session b's SOFC-Exp CRF run of seed 0
        sofc_repeat = ['--granularity', 'sofc', '--published-split', '--seeds', '0', '--crf',
                       '--tag', 'smokeaf', '--save-predictions', '--threads', str(THREADS)]
        return [[repeat + ['--tag', 'smokeae'], sofc_repeat,
                 *(sofc_split_job('random_split_vary', off) for off in OFFSETS_PUBLISHED)],
                [sofc_split_job('document_folds', off) for off in OFFSETS_PUBLISHED]]
    if name == 'w8b':
        # week 8, part 2: three papers at the size of two (one lane per granularity, so the
        # jobs that write a file share a lane)
        lanes = [[job(g, 'leave_one_paper_out', '--two-paper-size', offset=off, tag=f'm{off}')
                  for off in OFFSETS] for g in GRANULARITIES]
        return [[repeat + ['--tag', 'smokeag']] + lanes[0], lanes[1]]
    if name == 'w7a':
        # week 7, part 2a: recall calibration. An inner job is the 6 folds of a seed, each
        # trained on two papers (about 4 CRF runs); no predictions, the counts suffice
        bias = ('--entity-bias', *BIAS_GRID)
        inner = [(COST['crf'] * 4,
                  ['--granularity', g, '--leave-two-papers-out', '--crf', *bias, '--tag',
                   f'm{off}', *(('--seed-offset', str(off)) if off else ()), '--threads',
                   str(THREADS)])
                 for g in GRANULARITIES for off in OFFSETS]
        outer = [(COST['crf'] * RUNS['leave_one_paper_out'],
                  job(g, 'leave_one_paper_out', *bias, offset=off, tag=f'm{off}'))
                 for g in GRANULARITIES for off in OFFSETS + OFFSETS_W5]
        sofc = [(COST['sofc'] * len(OFFSETS_PUBLISHED), published_job('sofc', '--crf', *bias))]
        lanes = balance(inner + outer + sofc)
        return [[repeat + ['--tag', 'smokeu']] + lanes[0], lanes[1]]
    raise ValueError(f'unknown batch {name!r}')


BATCHES = ('smoke1', 'smoke2', 'smoke3', 'item1', 'item2', 'item12', 'item3', 'item3r',
           'smoke4', 'w5a', 'w5b', 'w5c', 'smoke5', 'smoke6', 'w6a', 'w6b', 'w7a', 'w7b',
           'w7c', 'w7d', 'w7e', 'w8a', 'w8b')


def batch_extras(name):
    """The extra pins of a batch (BATCH_EXTRAS; DeBERTa's for a stage-2 batch with it)."""
    if name.startswith('w6c:') and any('deberta' in m for m in stage2_methods(name)):
        return DEBERTA_PINS
    return BATCH_EXTRAS.get(name, {})


def batch_name(name):
    """argparse type of --batch: a name of BATCHES or a stage-2 batch w6c:M1,M2,..."""
    if name in BATCHES:
        return name
    if name.startswith('w6c:'):
        try:
            stage2_methods(name)
            return name
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    raise argparse.ArgumentTypeError(f'unknown batch {name!r}; known: {", ".join(BATCHES)}, '
                                     'w6c:<methods>')


def batch_spec(name):
    """The job spec that run_jobs.py runs (see its docstring), with the batch's extra pins;
    a batch whose jobs use the SOFC corpus also has its SHA-256 (sofc_corpus_sha256)."""
    extra = batch_extras(name)
    lanes = batch_lanes(name)
    spec = {'batch': name, 'gpu': GPU, 'pip': list(PIP) + list(extra.get('pip', ())),
            'wheels': list(WHEEL_FILES), 'models': {**MODELS, **extra.get('models', {})},
            'corpus_sha256': corpus_sha256(), 'deadline_hours': DEADLINE_HOURS,
            'lanes': lanes}
    if any(SOFC_CORPUS in job for lane in lanes for job in lane):
        spec['sofc_corpus_sha256'] = corpus_sha256(SOFC_CORPUS_REPORT)
    return spec


def render_script(spec, template=None):
    """run_jobs.py with JOBS set to `spec` (a JSON string literal, parsed on Kaggle)."""
    text = RUN_JOBS.read_text() if template is None else template
    if text.count(PLACEHOLDER) != 1:
        raise ValueError(f'{RUN_JOBS}: the JOBS placeholder line is missing or repeated')
    return text.replace(PLACEHOLDER, f'JOBS = json.loads({json.dumps(json.dumps(spec))})')


def kernel_metadata(user, kernel=KERNEL, dataset=DATASET):
    """kernel-metadata.json of a job kernel: private, GPU (T4 x2), internet, the bundle, and
    the pinned Kaggle image."""
    return {'id': f'{user}/{kernel}', 'title': kernel, 'code_file': 'run_jobs.py',
            'language': 'python', 'kernel_type': 'script', 'is_private': True,
            'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
            'dataset_sources': [f'{user}/{dataset}'], 'competition_sources': [],
            'kernel_sources': [], 'model_sources': [], 'machine_shape': MACHINE,
            'docker_image': DOCKER_IMAGE, 'docker_image_pinning_type': 'original'}


def dataset_metadata(user, dataset=DATASET):
    return {'title': dataset, 'id': f'{user}/{dataset}', 'licenses': [{'name': 'CC0-1.0'}]}


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------
def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def corpus_sha256(report=CORPUS_REPORT):
    """The SHA-256 of the fatigue corpus recorded when it was built (week 3), or of the
    corpus of another report (the SOFC corpus: SOFC_CORPUS_REPORT)."""
    return json.loads(Path(report).read_text())['corpus']['sha256']


def is_gpu_runs_file(path):
    """A results file of GPU runs: it has runs, and every run has an environment."""
    runs = json.loads(Path(path).read_text()).get('runs', [])
    return bool(runs) and all('environment' in r for r in runs)


def build_bundle(commit, out_dir, root=ROOT, results_dir=GPU_RESULTS, corpus=None,
                 expected_sha256=None, wheels=(), sofc_corpus=None, sofc_report=None):
    """Write the bundle files to out_dir (emptied first); returns BUNDLE.json's content.

    code.tar.gz is `git archive` of `commit` (the frozen code, tracked files only); the
    corpus must have the recorded SHA-256, and so must the SOFC corpus, which goes in as
    sofc_sentences.txt when its report (sofc_report, default SOFC_CORPUS_REPORT) exists
    (BUNDLE.json's sofc_corpus_sha256, else None); the results files of GPU runs go in as
    gpu__<name>.json, those whose every run has an environment (the votes made from GPU
    runs have none and are not needed to resume runs), and the `wheels` (paths) as they
    are.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    full = subprocess.run(['git', '-C', str(root), 'rev-parse', '--verify', f'{commit}^{{commit}}'],
                          capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(['git', '-C', str(root), 'archive', '--format=tar.gz', '-o',
                    str(out_dir / 'code.tar.gz'), full], check=True)
    corpus = Path(corpus) if corpus is not None else Path(root) / CORPUS
    expected = expected_sha256 or corpus_sha256()
    digest = file_sha256(corpus)
    if digest != expected:
        raise SystemExit(f'{corpus}: SHA-256 {digest}, not the recorded {expected}')
    shutil.copy2(corpus, out_dir / 'fatigue_sentences.txt')
    sofc_report = Path(sofc_report) if sofc_report is not None else SOFC_CORPUS_REPORT
    sofc_digest = None
    if sofc_report.exists():
        sofc_corpus = Path(sofc_corpus) if sofc_corpus is not None else Path(root) / SOFC_CORPUS
        if not sofc_corpus.is_file():
            raise SystemExit(f'{sofc_corpus} is missing (python src/graph/build_sofc_corpus.py '
                             f'rebuilds it, with --offline from its cache)')
        sofc_digest = file_sha256(sofc_corpus)
        if sofc_digest != corpus_sha256(sofc_report):
            raise SystemExit(f'{sofc_corpus}: SHA-256 {sofc_digest}, not the recorded '
                             f'{corpus_sha256(sofc_report)}')
        shutil.copy2(sofc_corpus, out_dir / 'sofc_sentences.txt')
    results = sorted(p.name for p in Path(results_dir).glob('ner_results_*.json')
                     if is_gpu_runs_file(p)) if Path(results_dir).is_dir() else []
    for name in results:
        shutil.copy2(Path(results_dir) / name, out_dir / f'{RESULTS_PREFIX}{name}')
    for wheel in wheels:
        shutil.copy2(wheel, out_dir / Path(wheel).name)
    info = {'commit': full, 'corpus_sha256': digest, 'sofc_corpus_sha256': sofc_digest,
            'results': results,
            'wheels': sorted(Path(w).name for w in wheels),
            'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (out_dir / 'BUNDLE.json').write_text(json.dumps(info, indent=2))
    return info


# ---------------------------------------------------------------------------
# Collecting results
# ---------------------------------------------------------------------------
ENV_KEYS = ('device', 'cuda', 'torch', 'transformers', 'deterministic')


def environment_key(run):
    env = run.get('environment')
    return None if env is None else tuple(env.get(k) for k in ENV_KEYS)


def check_collect(new, old=None, environments=()):
    """Problems with copying a pulled results file (`new`) over `old` in results/gpu/.

    Every run needs an environment, all the same and the same as `environments` (those
    of results/gpu/), and `old`'s runs must all be in `new`, unchanged.
    """
    problems = []
    runs = new.get('runs', [])
    keys = {environment_key(r) for r in runs}
    if None in keys:
        problems.append('runs without an environment (not GPU runs)')
    keys.discard(None)
    known = set(environments)
    if len(keys | known) > 1:
        problems.append(f'more than one environment: {sorted(map(str, keys | known))}')
    if old is not None:
        mine = {r.get('run_id'): r for r in runs}
        for r in old.get('runs', []):
            if mine.get(r.get('run_id')) != r:
                problems.append(f"run {r.get('run_id')} of the file in results/gpu is missing "
                                f"or changed")
    return problems


def collect(pulled, dest=GPU_RESULTS):
    """Copy the non-smoke results files of a pulled output to `dest`; returns the names."""
    sys.path.insert(0, str(ROOT / 'src' / 'graph'))
    from ner_baseline import parse_result_name
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(p.read_text()) for p in dest.glob('ner_results_*.json')]
    environments = {environment_key(r) for f in existing for r in f.get('runs', [])} - {None}
    copied, problems = [], {}
    for path in sorted(Path(pulled).rglob('ner_results_*.json')):
        parsed = parse_result_name(path.name)
        if parsed is None or 'smoke' in parsed[2]:
            continue
        new = json.loads(path.read_text())
        target = dest / path.name
        old = json.loads(target.read_text()) if target.exists() else None
        found = check_collect(new, old, environments)
        if found:
            problems[path.name] = found
            continue
        shutil.copy2(path, target)
        copied.append(path.name)
    return copied, problems


# ---------------------------------------------------------------------------
# The Kaggle command-line tool
# ---------------------------------------------------------------------------
def kaggle(*args, check=True):
    """Run the kaggle CLI (KAGGLE_CLI or `kaggle`); returns its output."""
    cli = os.environ.get('KAGGLE_CLI', 'kaggle')
    proc = subprocess.run([cli, *args], capture_output=True, text=True)
    out = (proc.stdout + proc.stderr).strip()
    if check and proc.returncode:
        raise SystemExit(f'kaggle {" ".join(args)} failed ({proc.returncode}): {out}')
    return out


def username(args):
    user = args.user or os.environ.get('KAGGLE_USERNAME')
    if not user:
        raise SystemExit('the Kaggle account name is needed: --user NAME or KAGGLE_USERNAME')
    return user


def build_wheels(out_dir=WORK_DIR / 'wheels'):
    """Pure-Python wheels of WHEELS (pip wheel --no-deps); returns their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', '-q', '--no-deps', '-w', str(out_dir),
                    *WHEELS], check=True)
    paths = [out_dir / name for name in WHEEL_FILES]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f'pip wheel did not make {missing}')
    return paths


def cmd_bundle(args):
    user = username(args)
    folder = WORK_DIR / 'bundle'
    info = build_bundle(args.commit, folder, wheels=build_wheels())
    (folder / 'dataset-metadata.json').write_text(json.dumps(dataset_metadata(user), indent=2))
    print(f"bundle: commit {info['commit'][:12]}, corpus {info['corpus_sha256'][:12]}, "
          f"{len(info['results'])} GPU results files")
    ref = f'{user}/{DATASET}'
    exists = 'ready' in kaggle('datasets', 'status', ref, check=False).lower()
    if exists:
        print(kaggle('datasets', 'version', '-p', str(folder), '-m',
                     f"commit {info['commit'][:12]}, {len(info['results'])} results files"))
    else:
        print(kaggle('datasets', 'create', '-p', str(folder)))
    for _ in range(60):              # Kaggle processes a new version before kernels see it
        time.sleep(10)
        status = kaggle('datasets', 'status', ref, check=False)
        if 'ready' in status.lower():
            print(f'{ref}: ready')
            return
    raise SystemExit(f'{ref} is not ready after 10 minutes: {status}')


def cmd_push(args):
    user = username(args)
    spec = batch_spec(args.batch)
    with tempfile.TemporaryDirectory() as folder:
        Path(folder, 'run_jobs.py').write_text(render_script(spec))
        Path(folder, 'kernel-metadata.json').write_text(
            json.dumps(kernel_metadata(user, args.kernel), indent=2))
        timeout = str(int(3600 * (DEADLINE_HOURS + 1.5)))
        out = kaggle('kernels', 'push', '-p', folder, '-t', timeout)
        print(out)
    if 'error' in out.lower():       # the CLI exits 0 on a refused push (e.g. 2 GPU sessions)
        raise SystemExit(f'batch {args.batch} not pushed')
    print(f"batch {args.batch}: " + '; '.join(f'lane {i}: {len(jobs)} jobs'
                                              for i, jobs in enumerate(spec['lanes'])))


def cmd_status(args):
    print(kaggle('kernels', 'status', f'{username(args)}/{args.kernel}'))


def cmd_pull(args):
    out = WORK_DIR / args.kernel / time.strftime('%Y%m%dT%H%M%S')
    out.mkdir(parents=True)
    print(kaggle('kernels', 'output', f'{username(args)}/{args.kernel}', '-p', str(out)))
    manifest = out / 'manifest.json'
    if not manifest.exists():
        print(f'{out}: no manifest.json')
        return
    data = json.loads(manifest.read_text())
    setup = data.get('setup', {})
    if setup.get('error'):
        print('setup error:\n' + setup['error'])
    print(f"bundle {setup.get('bundle', {}).get('commit', '?')[:12]}, GPUs {setup.get('gpus')}, "
          f"setup {setup.get('setup_minutes')} min, total {data.get('total_minutes')} min")
    for j in data.get('jobs', []):
        print(f"  lane {j['lane']} job {j['index']}: {j['status']} "
              f"{j.get('minutes', '')} min  {' '.join(j['args'])}")
    print(f'pulled to {out}')


def cmd_collect(args):
    copied, problems = collect(args.pulled)
    for name in copied:
        print(f'  copied {name}')
    for name, found in problems.items():
        print(f"  NOT copied {name}: {'; '.join(found)}")
    print(f'{len(copied)} files copied to {GPU_RESULTS}, {len(problems)} refused')


def cmd_quota(args):
    print(kaggle('quota'))


def main(argv=None):
    parser = argparse.ArgumentParser(description='NER runs on Kaggle GPUs (week-4 protocol)')
    parser.add_argument('--user', default=None, help='Kaggle account (default: KAGGLE_USERNAME)')
    parser.add_argument('--kernel', default=KERNEL,
                        help=f'the job kernel (default: {KERNEL}); another one runs a second '
                             'session at the same time')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('bundle', help='build and upload the bundle dataset')
    p.add_argument('--commit', default='HEAD', help='the frozen commit (default: HEAD)')
    p.set_defaults(func=cmd_bundle)
    p = sub.add_parser('push', help="push a batch's jobs as a GPU kernel")
    p.add_argument('--batch', type=batch_name, required=True,
                   help=f"one of {', '.join(BATCHES)}, or w6c:<methods> (stage 2 of week 6)")
    p.set_defaults(func=cmd_push)
    sub.add_parser('status', help="the kernel's status").set_defaults(func=cmd_status)
    sub.add_parser('pull', help="download the kernel's outputs").set_defaults(func=cmd_pull)
    p = sub.add_parser('collect', help='copy pulled results files to results/gpu/')
    p.add_argument('pulled', help='a directory made by pull')
    p.set_defaults(func=cmd_collect)
    sub.add_parser('quota', help='weekly GPU quota').set_defaults(func=cmd_quota)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
