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
Week 9 (mention replacement on the multi-task model, --multi-task --augment-mentions 3):
  w9a     the CRF run of smoke4 (smokeah) and the multi-task run of random split 0 (smokeai)
          again; then crf_mt_mr_m<N> on the held-out papers, seed offsets 0 to 900 (both
          granularities from one model), and S2 on SOFC-Exp's two levels (sofc_both, model
          seeds 0 to 900), balanced over the lanes
  w9b     the CRF run of smoke4 (smokeaj) and session b's SOFC-Exp CRF run of seed 0 (smokeak)
          again; then SOFC-Exp's entity level trained on 10 of its 26 training articles
          (--train-articles 10, 5 draws), the CRF head and mention replacement, seed offsets 0
          to 900 (crf_m<N>, crf_mr_m<N>), balanced over the lanes
Week 10 (MuLMS, a third corpus: mulms.py; its outermost entities):
  w10a    the CRF run of smoke4 (smokeal); then on MuLMS's 7 held-out subfields
          (--leave-one-subfield-out) the CRF head and mention replacement, and on its 5
          random pool splits the CRF head, each trained on 400 sentences (--train-pool 400),
          seed offsets 0 to 900 (crf_m<N>, crf_mr_m<N>), balanced over the lanes; the bundle
          holds the converted corpus (its SHA-256, mulms_sha256)
Week 11 (mention replacement with silver names: --augment-names; 3 teachers, 10 students):
  w11a    the CRF run of smoke4 (smokeam); then on MuLMS's 7 held-out subfields (one job
          each, --subfields), 3 CRF teachers tag the fold's sentences outside its pool and
          test set (--self-train unused) and the students train mention replacement whose
          pool holds their silver names (crf_mrs_m<N>). Each lane trains the same teachers
          and half of the students (seed offsets 0-400 and 500-900), so their files differ
  w11b    the CRF run of smoke4 (smokean); then on MaterioMiner's 4 held-out papers (one job
          each, --papers), the teachers tag the whole fatigue corpus; one lane per
          granularity (crf_mrs_m<N>)
Week 12 (names from a topic corpus: build_topic_corpus.py, one per MuLMS subfield, in the
bundle as topic_<topic>_sentences.txt; the screen, then the decision for its winner):
  w12a    the CRF run of smoke4 (smokeao); then the screen, seed offsets 0 to 200, on MuLMS's
          7 held-out subfields, five variants (amendment 1), per subfield one job in which
          the 3 teachers tag the subfield's topic corpus: in lane 0 the students train
          mention replacement with the silver names of the sentence filter
          (crf_mrs_topic_m<N>) and, with the mention-level filter (--silver-mentions) and 6
          copies, crf_mrm_topic6_m<N>; in lane 1 the mention-level filter with 3 copies
          (crf_mrm_topic_m<N>) and self-training on the silver sentences (crf_st3_topic_m<N>);
          and mention replacement with 6 copies and no names (crf_mr6_m<N>, one job per seed
          offset, no teachers), one job in lane 0 and two in lane 1. The jobs of a variant
          share a lane: they extend the same files
  w12b:V  the decision for the screen's winner V (mrs, mrm, mrm6, st3 or mr6): seed offsets
          300 to 900; the teacher variants as in w12a with each lane training the teachers
          and 3 or 4 of the students (offsets 300-500 and 600-900), mr6 one job per offset
          over the lanes (smokeap)

The TPU spike (docs/METHODS_PLAN.md §5; every batch named tpu... runs on the TPU v5e-8, one
lane per chip, its jobs with --device xla, its results in results/tpu/, a family of their own):
  tpu-smoke1  one fine-grained CRF job (the 4 held-out papers, seed 0) on one chip (crf_tpu1):
          does the XLA path run on the TPU, and how long is a run
  tpu-smoke  the same job on all 8 chips at once (crf_tpua to crf_tpuh): the wall time per
          run on one chip while 8 run, and whether the 8 files are identical

Usage:
  python src/graph/kaggle_jobs.py quota
  python src/graph/kaggle_jobs.py bundle --commit HEAD
  python src/graph/kaggle_jobs.py push --batch smoke1
  python src/graph/kaggle_jobs.py --dataset ner-gpu-bundle-2 bundle --commit HEAD   # a second
  python src/graph/kaggle_jobs.py --dataset ner-gpu-bundle-2 --kernel ner-gpu-jobs-b push ...
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
# the TPU spike (docs/METHODS_PLAN.md §5): a batch named tpu... runs on Kaggle's TPU v5e-8, one
# lane per chip, with --device xla; its results are a family of their own (results/tpu/)
TPU_LANES = 8
TPU_DEADLINE_HOURS = 7              # a Kaggle TPU session lasts 9 hours
TPU_RESULTS = ROOT / 'results' / 'tpu'
# Kaggle's TPU VM image (19 October: torch 2.11.0+cpu, no torch_xla): torch_xla and the
# matching torch (the CPU wheel) with the TPU runtime, installed by the runner with PIP
# (torch_xla[tpu] pins libtpu 0.0.21, which PyPI no longer has: its 0.0.21.1 is pinned here;
# torchaudio and torchvision follow torch, else the image's builds for its torch break
# transformers' import)
TPU_PINS = {'pip': ('--extra-index-url', 'https://download.pytorch.org/whl/cpu',
                    'torch==2.9.0+cpu', 'torchaudio==2.9.0+cpu', 'torchvision==0.24.0+cpu',
                    'torch_xla==2.9.0', 'libtpu==0.0.21.1')}
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
        'mr': 3.8,           # mention replacement, 3 copies: about 4 times the training sentences
        'mr6': 6.6}          # 6 copies: about 7 times
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
# week 9, part 2: 5 draws of 10 of SOFC-Exp's 26 training articles, about 0.35 of its official
# training sentences per run
DRAWS = 5
DRAW_SHARE = 0.35
# week 10: MuLMS (mulms.py), converted locally into MULMS_DIR (CC BY-SA 4.0, not in git) and
# shipped in the bundle; its report records the files' SHA-256. A run trains on MULMS_POOL
# sentences; its 7 held-out subfields test every sentence once per job
MULMS_DIR = 'data/processed/mulms'
MULMS_REPORT = ROOT / 'results' / 'mulms_corpus.json'
MULMS_POOL = 400
MULMS_SUBFIELDS = 7
COST['mulms'] = 1.5         # a run on 400 MuLMS sentences of 29 words, and a larger test
MULMS_PREFIX = 'mulms__'    # the bundle names of the MuLMS files (as in run_jobs.py)
# week 12: an unlabeled corpus per MuLMS subfield (build_topic_corpus.py), each with a report
# that records its SHA-256; a batch whose jobs read one ships it as topic_<topic>_sentences.txt
TOPIC_NAMES = ('electrolysis', 'graphene', 'pemfc', 'poly', 'semi', 'sofc', 'steel')
TOPIC_CORPUS_DIR = 'data/processed/corpus'
TOPIC_CORPUS = TOPIC_CORPUS_DIR + '/topic_{topic}_sentences.txt'
TOPIC_REPORTS = ROOT / 'results' / 'topic_corpora'
# week 11: the subfields (MuLMS) and the students of each lane (both lanes train the teachers)
MULMS_SUBFIELD_NAMES = ('electrolysis', 'graphene', 'pemfc', 'poly', 'semi', 'sofc', 'steel')
MRS_LANES = (OFFSETS_PUBLISHED[:5], OFFSETS_PUBLISHED[5:])
MRS_FLAGS = ('--crf', '--teachers', '3', '--students', '10', '--augment-mentions', '3',
             '--augment-names', '--save-predictions', '--threads', str(THREADS))
TOPIC_TAG = 'topic'          # --names-tag of the topic-corpus variants (crf_mrs_topic, crf_mrm_topic)
MR6_COPIES = 6               # week 12's copy-count variant (crf_mr6)
# the screen's variants (amendment 1 added mrm6 and st3): w12b:<variant> decides one
WEEK12_VARIANTS = ('mrs', 'mrm', 'mr6', 'mrm6', 'st3')
DECISION_LANES = (OFFSETS_W5[:3], OFFSETS_W5[3:])   # w12b: the students per lane (seeds 300-900)


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


def mrs_mulms_job(subfield, student_offsets):
    """Week 11 on MuLMS: one held-out subfield, 400 pool sentences; the 3 teachers tag the
    fold's unused sentences and the students of `student_offsets` (of 10) train mention
    replacement with their silver names (crf_mrs_m<N>)."""
    return ['--granularity', 'mulms', '--leave-one-subfield-out', '--train-pool',
            str(MULMS_POOL), '--subfields', subfield, '--self-train', 'unused',
            '--student-offsets', *(str(o) for o in student_offsets), *MRS_FLAGS]


def topic_mulms_job(subfield, student_offsets, mentions=False, copies=3, names=True):
    """Week 12 on MuLMS: one held-out subfield, 400 pool sentences; the 3 teachers tag the
    subfield's topic corpus (TOPIC_CORPUS, in the bundle; all of it) and the students of
    `student_offsets` train mention replacement with its silver names (crf_mrs_topic_m<N>;
    with `mentions`, the mention-level silver filter: crf_mrm_topic_m<N>; `copies` of each
    sentence, 6 for crf_mrm_topic6). Without `names` the students self-train on the silver
    sentences instead (week 4's method, 3 teachers: crf_st3_topic_m<N>)."""
    job = ['--granularity', 'mulms', '--leave-one-subfield-out', '--train-pool',
           str(MULMS_POOL), '--subfields', subfield, '--self-train',
           TOPIC_CORPUS.format(topic=subfield), '--self-train-sample', '0',
           '--student-offsets', *(str(o) for o in student_offsets), '--names-tag', TOPIC_TAG]
    if not names:
        return job + ['--crf', '--teachers', '3', '--students', '10', '--save-predictions',
                      '--threads', str(THREADS)]
    flags = [f if f != '3' or MRS_FLAGS[i - 1] != '--augment-mentions' else str(copies)
             for i, f in enumerate(MRS_FLAGS)]
    return job + (['--silver-mentions'] if mentions else []) + flags


def topic_variant_job(variant, subfield, student_offsets):
    """The job of a week-12 names or self-training variant (WEEK12_VARIANTS but mr6)."""
    return topic_mulms_job(subfield, student_offsets, mentions=variant in ('mrm', 'mrm6'),
                           copies=6 if variant == 'mrm6' else 3, names=variant != 'st3')


def mr6_mulms_job(offset):
    """Week 12 on MuLMS: mention replacement with MR6_COPIES copies (crf_mr6_m<N>) on the 7
    held-out subfields, 400 pool sentences, one job per seed offset."""
    return ['--granularity', 'mulms', '--leave-one-subfield-out', '--train-pool',
            str(MULMS_POOL), '--crf', '--augment-mentions', str(MR6_COPIES), '--tag',
            f'm{offset}', *(('--seed-offset', str(offset)) if offset else ()),
            '--save-predictions', '--threads', str(THREADS)]


def week12_variant(name):
    """The variant of a decision batch name 'w12b:V' (ValueError if unknown)."""
    variant = name[len('w12b:'):]
    if variant not in WEEK12_VARIANTS:
        raise ValueError(f'unknown week-12 variant {name!r}; known: {WEEK12_VARIANTS}')
    return variant


def mrs_paper_job(granularity, paper):
    """Week 11 on MaterioMiner: one held-out paper; the 3 teachers tag the whole fatigue
    corpus and the 10 students train mention replacement with their silver names."""
    return ['--granularity', granularity, '--leave-one-paper-out', '--papers', paper,
            '--self-train', CORPUS, '--self-train-sample', '0', *MRS_FLAGS]


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
    if name == 'w9a':
        # week 9: mention replacement on the multi-task model (crf_mt_mr), one job per seed
        # offset on the held-out papers (both granularities from one model), and S2 on
        # SOFC-Exp's two levels, model seeds 0 to 900 (one job, a file per level); first the
        # repeats of the CRF run of smoke4 (smokeah) and of the multi-task run of random
        # split 0 (smokeai: the multi-task code gained mention replacement)
        mt_repeat = ['--vary-split', '--seeds', '0', '--crf', '--multi-task', '--tag',
                     'smokeai', '--save-predictions', '--threads', str(THREADS)]
        held_out = [(COST['mt'] * COST['mr'] * RUNS['leave_one_paper_out'],
                     ['--leave-one-paper-out', '--crf', '--multi-task', *METHOD_FLAGS['mr'],
                      '--tag', f'm{off}', *(('--seed-offset', str(off)) if off else ()),
                      '--save-predictions', '--threads', str(THREADS)])
                    for off in OFFSETS_PUBLISHED]
        sofc = [(COST['sofc'] * COST['mt'] * COST['mr'] * len(OFFSETS_PUBLISHED),
                 published_job('sofc_both', '--crf', '--multi-task', *METHOD_FLAGS['mr']))]
        lanes = balance(held_out + sofc, loads=[COST['crf'], COST['mt']])
        return [[repeat + ['--tag', 'smokeah']] + lanes[0], [mt_repeat] + lanes[1]]
    if name == 'w9b':
        # week 9, part 2: SOFC-Exp's entity level trained on 10 of its 26 training articles
        # (--train-articles 10, 5 draws; the official validation and test articles), the CRF
        # head and mention replacement, one job per system and seed offset (a file each);
        # first the repeats of the CRF run of smoke4 (smokeaj) and of session b's SOFC-Exp CRF
        # run of seed 0 (smokeak: this commit changed how the official split is cut)
        sofc_repeat = ['--granularity', 'sofc', '--published-split', '--seeds', '0', '--crf',
                       '--tag', 'smokeak', '--save-predictions', '--threads', str(THREADS)]
        draws = [(COST['sofc'] * DRAW_SHARE * DRAWS * cost,
                  ['--granularity', 'sofc', '--published-split', '--train-articles', '10',
                   '--crf', *flags, '--tag', f'm{off}',
                   *(('--seed-offset', str(off)) if off else ()), '--save-predictions',
                   '--threads', str(THREADS)])
                 for flags, cost in (((), 1.0), (METHOD_FLAGS['mr'], COST['mr']))
                 for off in OFFSETS_PUBLISHED]
        lanes = balance(draws, loads=[COST['crf'] + COST['sofc'], 0.0])
        return [[repeat + ['--tag', 'smokeaj'], sofc_repeat] + lanes[0], lanes[1]]
    if name == 'w10a':
        # week 10: MuLMS, a third corpus; per seed offset one job of the CRF head and one of
        # mention replacement on the 7 held-out subfields, and one of the CRF head on the 5
        # random pool splits (a file each); first the repeat of the CRF run of smoke4 (smokeal)
        held = ('--granularity', 'mulms', '--leave-one-subfield-out', '--train-pool',
                str(MULMS_POOL))
        pool = ('--granularity', 'mulms', '--vary-split', '--train-pool', str(MULMS_POOL),
                '--seeds', *(str(k) for k in range(5)))
        jobs = [(COST['mulms'] * runs * cost,
                 [*protocol, '--crf', *flags, '--tag', f'm{off}',
                  *(('--seed-offset', str(off)) if off else ()), '--save-predictions',
                  '--threads', str(THREADS)])
                for protocol, runs, flags, cost in ((held, MULMS_SUBFIELDS, (), 1.0),
                                                    (held, MULMS_SUBFIELDS,
                                                     METHOD_FLAGS['mr'], COST['mr']),
                                                    (pool, 5, (), 1.0))
                for off in OFFSETS_PUBLISHED]
        lanes = balance(jobs, loads=[COST['crf'], 0.0])
        return [[repeat + ['--tag', 'smokeal']] + lanes[0], lanes[1]]
    if name == 'w11a':
        # week 11 on MuLMS: both lanes train the teachers of every subfield, each lane half of
        # the students, so that no two lanes write the same file
        return [[repeat + ['--tag', 'smokeam']] +
                [mrs_mulms_job(s, MRS_LANES[0]) for s in MULMS_SUBFIELD_NAMES],
                [mrs_mulms_job(s, MRS_LANES[1]) for s in MULMS_SUBFIELD_NAMES]]
    if name == 'tpu-smoke1':
        # the TPU spike, first session: one fine-grained CRF job (the 4 held-out papers, seed
        # 0) on one chip, as crf_tpu1: does the XLA path run on the TPU, and how long is a run
        return [[job('fine', 'leave_one_paper_out', '--device', 'xla', tag='tpu1')]]
    if name == 'tpu-smoke':
        # the TPU spike, second session: the same job on every chip at once, as crf_tpua to
        # crf_tpuh: the wall time per run on one chip while 8 run, and whether the 8 files
        # are identical (and equal to crf_tpu1's)
        return [[job('fine', 'leave_one_paper_out', '--device', 'xla', tag=f'tpu{letter}')]
                for letter in 'abcdefgh'[:TPU_LANES]]
    if name == 'w11b':
        # week 11 on MaterioMiner: one lane per granularity, a job per held-out paper
        return [[repeat + ['--tag', 'smokean']] + [mrs_paper_job('fine', p) for p in PAPERS],
                [mrs_paper_job('coarse', p) for p in PAPERS]]
    if name == 'w12a':
        # week 12's screen on MuLMS, seed offsets 0 to 200 (amendment 1: five variants): the
        # jobs of a variant share a lane (they extend the same files); lane 0 the sentence
        # filter and the mention filter with 6 copies, lane 1 the mention filter and
        # self-training; the 6-copy jobs without names, a file each, fill the lanes
        return [[repeat + ['--tag', 'smokeao']]
                + [topic_variant_job(v, s, OFFSETS) for v in ('mrs', 'mrm6')
                   for s in MULMS_SUBFIELD_NAMES]
                + [mr6_mulms_job(OFFSETS[0])],
                [topic_variant_job(v, s, OFFSETS) for v in ('mrm', 'st3')
                 for s in MULMS_SUBFIELD_NAMES]
                + [mr6_mulms_job(off) for off in (OFFSETS[1], OFFSETS[2])]]
    if name.startswith('w12b:'):
        # week 12's decision on MuLMS for the screen's winner, seed offsets 300 to 900
        variant = week12_variant(name)
        if variant == 'mr6':
            lanes = balance([(COST['mulms'] * MULMS_SUBFIELDS * COST['mr6'], mr6_mulms_job(off))
                             for off in OFFSETS_W5], loads=[COST['crf'], 0.0])
        else:
            lanes = [[topic_variant_job(variant, s, offsets) for s in MULMS_SUBFIELD_NAMES]
                     for offsets in DECISION_LANES]
        return [[repeat + ['--tag', 'smokeap']] + lanes[0], lanes[1]]
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
           'w7c', 'w7d', 'w7e', 'w8a', 'w8b', 'w9a', 'w9b', 'w10a', 'w11a', 'w11b', 'w12a',
           *(f'w12b:{v}' for v in WEEK12_VARIANTS), 'tpu-smoke1', 'tpu-smoke')


def batch_extras(name):
    """The extra pins of a batch (BATCH_EXTRAS; DeBERTa's for a stage-2 batch with it;
    torch_xla's for a TPU batch)."""
    if name.startswith('w6c:') and any('deberta' in m for m in stage2_methods(name)):
        return DEBERTA_PINS
    if is_tpu_batch(name):
        return TPU_PINS
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
    if any(job[i:i + 2] == ['--granularity', 'mulms'] for lane in lanes for job in lane
           for i in range(len(job) - 1)):
        spec['mulms_files'] = mulms_files()
    topics = sorted(t for t in TOPIC_NAMES
                    if any(TOPIC_CORPUS.format(topic=t) in job for lane in lanes for job in lane))
    if topics:
        spec['topic_corpora'] = {t: corpus_sha256(TOPIC_REPORTS / f'{t}.json') for t in topics}
    if is_tpu_batch(name):
        # TPU results are a family of their own: no GPU results file is restored (it would
        # only cost setup time; a run would not be skipped for it either, its tag differs)
        spec.update({'gpu': None, 'device': 'xla', 'deadline_hours': TPU_DEADLINE_HOURS,
                     'restore_results': False})
    return spec


def render_script(spec, template=None):
    """run_jobs.py with JOBS set to `spec` (a JSON string literal, parsed on Kaggle)."""
    text = RUN_JOBS.read_text() if template is None else template
    if text.count(PLACEHOLDER) != 1:
        raise ValueError(f'{RUN_JOBS}: the JOBS placeholder line is missing or repeated')
    return text.replace(PLACEHOLDER, f'JOBS = json.loads({json.dumps(json.dumps(spec))})')


def kernel_metadata(user, kernel=KERNEL, dataset=DATASET, tpu=False):
    """kernel-metadata.json of a job kernel: private, GPU (T4 x2), internet, the bundle, and
    the pinned Kaggle image; with `tpu`, the TPU (v5e-8) and Kaggle's TPU environment instead
    of the GPU and the pinned image, and the script wrapped as a notebook (notebook_cell): four
    script sessions of 19 October asked for the TPU and ran on a plain CPU machine."""
    meta = {'id': f'{user}/{kernel}', 'title': kernel,
            'code_file': 'run_jobs.ipynb' if tpu else 'run_jobs.py',
            'language': 'python', 'kernel_type': 'notebook' if tpu else 'script',
            'is_private': True, 'enable_gpu': not tpu, 'enable_tpu': tpu,
            'enable_internet': True, 'dataset_sources': [f'{user}/{dataset}'],
            'competition_sources': [], 'kernel_sources': [], 'model_sources': []}
    if not tpu:
        meta.update({'machine_shape': MACHINE, 'docker_image': DOCKER_IMAGE,
                     'docker_image_pinning_type': 'original'})
    return meta


def notebook_cell(source):
    """A one-cell notebook (nbformat 4) holding `source` as its code cell, as JSON text."""
    return json.dumps({'nbformat': 4, 'nbformat_minor': 5,
                       'metadata': {'kernelspec': {'name': 'python3', 'language': 'python',
                                                   'display_name': 'Python 3'}},
                       'cells': [{'cell_type': 'code', 'execution_count': None, 'id': 'jobs',
                                  'metadata': {}, 'outputs': [], 'source': source}]}, indent=1)


def is_tpu_batch(name):
    """A batch that runs on the TPU (its name starts with tpu)."""
    return name.startswith('tpu')


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


def mulms_files(report=None):
    """{name: SHA-256} of the converted MuLMS files, as mulms.py recorded them (outputs) in
    its report (default MULMS_REPORT)."""
    report = Path(report) if report is not None else MULMS_REPORT
    if not report.exists():
        raise SystemExit(f'{report} is missing (python src/graph/mulms.py)')
    return dict(json.loads(report.read_text())['outputs'])


def corpus_sha256(report=CORPUS_REPORT):
    """The SHA-256 of the fatigue corpus recorded when it was built (week 3), or of the
    corpus of another report (the SOFC corpus: SOFC_CORPUS_REPORT)."""
    return json.loads(Path(report).read_text())['corpus']['sha256']


def is_gpu_runs_file(path):
    """A results file of GPU runs: it has runs, and every run has an environment."""
    runs = json.loads(Path(path).read_text()).get('runs', [])
    return bool(runs) and all('environment' in r for r in runs)


def build_bundle(commit, out_dir, root=ROOT, results_dir=GPU_RESULTS, corpus=None,
                 expected_sha256=None, wheels=(), sofc_corpus=None, sofc_report=None,
                 mulms_report=None, mulms_dir=None, topic_reports=None, topic_dir=None):
    """Write the bundle files to out_dir (emptied first); returns BUNDLE.json's content.

    code.tar.gz is `git archive` of `commit` (the frozen code, tracked files only); the
    corpus must have the recorded SHA-256, and so must the SOFC corpus, which goes in as
    sofc_sentences.txt when its report (sofc_report, default SOFC_CORPUS_REPORT) exists
    (BUNDLE.json's sofc_corpus_sha256, else None); the results files of GPU runs go in as
    gpu__<name>.json, those whose every run has an environment (the votes made from GPU
    runs have none and are not needed to resume runs), and the `wheels` (paths) as they
    are. When MuLMS's report (mulms_report, default MULMS_REPORT) exists, its converted
    files (mulms_dir, default MULMS_DIR) go in as mulms__<name>, each with its recorded
    SHA-256 (BUNDLE.json's mulms_files, else None). Every topic corpus with a report under
    topic_reports (default TOPIC_REPORTS; build_topic_corpus.py), read from topic_dir (default
    root's TOPIC_CORPUS_DIR), goes in as topic_<topic>_sentences.txt with its recorded SHA-256
    (BUNDLE.json's topic_corpora, else None).
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
    mulms_report = Path(mulms_report) if mulms_report is not None else MULMS_REPORT
    mulms = mulms_files(mulms_report) if mulms_report.exists() else None
    mulms_dir = Path(mulms_dir) if mulms_dir is not None else Path(root) / MULMS_DIR
    for name, recorded in (mulms or {}).items():
        path = mulms_dir / name
        if not path.is_file() or file_sha256(path) != recorded:
            raise SystemExit(f'{path} is missing or not the recorded file (python '
                             f'src/graph/mulms.py rebuilds it, with --offline from its cache)')
        shutil.copy2(path, out_dir / f'{MULMS_PREFIX}{name}')
    topic_reports = Path(topic_reports) if topic_reports is not None else TOPIC_REPORTS
    topic_dir = Path(topic_dir) if topic_dir is not None else Path(root) / TOPIC_CORPUS_DIR
    topics = {}
    for report in sorted(topic_reports.glob('*.json')) if topic_reports.is_dir() else []:
        topic = report.stem
        path = topic_dir / f'topic_{topic}_sentences.txt'
        if not path.is_file():
            raise SystemExit(f'{path} is missing (python src/graph/build_topic_corpus.py --topic '
                             f'{topic} rebuilds it, with --offline from its cache)')
        topics[topic] = file_sha256(path)
        if topics[topic] != corpus_sha256(report):
            raise SystemExit(f'{path}: SHA-256 {topics[topic]}, not the recorded '
                             f'{corpus_sha256(report)}')
        shutil.copy2(path, out_dir / f'topic_{topic}_sentences.txt')
    results = sorted(p.name for p in Path(results_dir).glob('ner_results_*.json')
                     if is_gpu_runs_file(p)) if Path(results_dir).is_dir() else []
    for name in results:
        shutil.copy2(Path(results_dir) / name, out_dir / f'{RESULTS_PREFIX}{name}')
    for wheel in wheels:
        shutil.copy2(wheel, out_dir / Path(wheel).name)
    info = {'commit': full, 'corpus_sha256': digest, 'sofc_corpus_sha256': sofc_digest,
            'mulms_files': mulms, 'topic_corpora': topics or None, 'results': results,
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
    (folder / 'dataset-metadata.json').write_text(
        json.dumps(dataset_metadata(user, args.dataset), indent=2))
    print(f"bundle: commit {info['commit'][:12]}, corpus {info['corpus_sha256'][:12]}, "
          f"{len(info['results'])} GPU results files")
    ref = f'{user}/{args.dataset}'
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
    tpu = is_tpu_batch(args.batch)
    with tempfile.TemporaryDirectory() as folder:
        script = render_script(spec)
        if tpu:
            Path(folder, 'run_jobs.ipynb').write_text(notebook_cell(script))
        else:
            Path(folder, 'run_jobs.py').write_text(script)
        Path(folder, 'kernel-metadata.json').write_text(
            json.dumps(kernel_metadata(user, args.kernel, args.dataset, tpu=tpu), indent=2))
        timeout = str(int(3600 * (spec['deadline_hours'] + 1.5)))
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
    dest = Path(args.results_dir)
    copied, problems = collect(args.pulled, dest)
    for name in copied:
        print(f'  copied {name}')
    for name, found in problems.items():
        print(f"  NOT copied {name}: {'; '.join(found)}")
    print(f'{len(copied)} files copied to {dest}, {len(problems)} refused')


def cmd_quota(args):
    print(kaggle('quota'))


def main(argv=None):
    parser = argparse.ArgumentParser(description='NER runs on Kaggle GPUs (week-4 protocol)')
    parser.add_argument('--user', default=None, help='Kaggle account (default: KAGGLE_USERNAME)')
    parser.add_argument('--kernel', default=KERNEL,
                        help=f'the job kernel (default: {KERNEL}); another one runs a second '
                             'session at the same time')
    parser.add_argument('--dataset', default=DATASET,
                        help=f'the bundle dataset (default: {DATASET}); another one holds a '
                             'second bundle, so that sessions from two commits can be queued '
                             'at the same time')
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
    p.add_argument('--results-dir', default=str(GPU_RESULTS), metavar='DIR',
                   help='where the files go (TPU batches: results/tpu)')
    p.set_defaults(func=cmd_collect)
    sub.add_parser('quota', help='weekly GPU quota').set_defaults(func=cmd_quota)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
