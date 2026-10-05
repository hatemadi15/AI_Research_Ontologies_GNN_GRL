"""
config.py - Configuration for the Ontology Learning Pipeline

All switches are read from environment variables so that every stage (each
runs as its own process, see run_pipeline.py) and every ablation preset (see
run_ablation.py) sees the same configuration:

  Component flags   USE_GNN_EMBEDDINGS, USE_BIDIRECTIONAL, USE_COMBINED_SCORING,
                    USE_HEARST_PATTERNS, USE_DEP_PARSING, CORPUS_AUGMENT
  Leakage control   ORACLE_TYPES (default false). When true, gold NER types are
                    allowed to influence predictions (type-match boost in
                    align.py, NER-type-guided clustering). Results produced this
                    way are an upper bound, never a model result.
  LLM features      LLM_ALIGNMENT, LLM_RELATIONS, RAG_TYPING plus the LLM_*
                    backend settings (OpenRouter or OpenAI)
  Paths / runs      PROCESSED_DIR, LLM_CACHE_PATH, SEED, EMBEDDER_MODEL
"""

import os
import random

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dataset")
DEFAULT_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
# Overridable so ablation presets write to separate directories
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", DEFAULT_PROCESSED_DIR)
# LLM answers are shared across runs/presets (keys include the model name)
LLM_CACHE_PATH = os.environ.get(
    "LLM_CACHE_PATH", os.path.join(DEFAULT_PROCESSED_DIR, "llm_cache.json")
)


def _env_flag(name, default):
    """Boolean environment variable ('1', 'true', 'yes', 'on' are true)."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return bool(default)
    return value.strip().lower() in ("1", "true", "yes", "on")


# Default embedding model. Override per preset or via EMBEDDER_MODEL.
DEFAULT_EMBEDDER_MODEL = 'all-MiniLM-L6-v2'
EMBEDDER_MODEL = os.environ.get("EMBEDDER_MODEL", DEFAULT_EMBEDDER_MODEL)

# Global seed for random / numpy / torch (and PyG's negative sampling, which
# uses Python's `random`)
SEED = int(os.environ.get("SEED", "0"))


def set_seed(seed=None):
    """Seed Python, NumPy and (if installed) PyTorch for reproducible runs."""
    seed = SEED if seed is None else seed
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# File names shared between stages
GOLD_TYPES_FILE = "gold_term_types.json"          # written by builder.py
PREDICTED_TYPES_BASE_FILE = "predicted_types_base.json"  # align.py
PREDICTED_TYPES_FILE = "predicted_types.json"     # rag_typing.py (final)

# ---- Leakage control ----
# Gold NER types may only influence predictions when ORACLE_TYPES is set.
ORACLE_TYPES = _env_flag("ORACLE_TYPES", False)
# NER-type-guided clustering groups terms by their gold type, so it is an
# oracle component and follows ORACLE_TYPES unless set explicitly.
NER_TYPE_CLUSTERING = _env_flag("NER_TYPE_CLUSTERING", ORACLE_TYPES)

# ---- Component flags (ablations) ----
USE_GNN_EMBEDDINGS = _env_flag("USE_GNN_EMBEDDINGS", True)
USE_BIDIRECTIONAL = _env_flag("USE_BIDIRECTIONAL", True)
USE_COMBINED_SCORING = _env_flag("USE_COMBINED_SCORING", True)
USE_HEARST_PATTERNS = _env_flag("USE_HEARST_PATTERNS", True)
USE_DEP_PARSING = _env_flag("USE_DEP_PARSING", True)

# Weight of graph-neighbour class scores when typing a term:
#   S' = (1 - beta) * S + beta * W @ S
# with W the row-normalised top-k GNN-similarity neighbour matrix. 0.3 keeps
# the original design's text/GNN fusion weight (alpha = 0.7 text).
GNN_SMOOTHING_BETA = float(os.environ.get("GNN_SMOOTHING_BETA", "0.3"))
GNN_SMOOTHING_K = int(os.environ.get("GNN_SMOOTHING_K", "5"))

# LLM validation mode for cluster naming / dep-parse relation typing
LLM_MODE = _env_flag("LLM_MODE", False)

# ---- LLM backend configuration ----
# The provider is chosen from whichever key is set, so a key is never sent to
# another provider's endpoint: OPENROUTER_API_KEY -> OpenRouter (preferred),
# otherwise OPENAI_API_KEY -> OpenAI. LLM_BASE_URL / LLM_MODEL override the
# provider defaults (e.g. to use another OpenAI-compatible endpoint or model).
_OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if _OPENROUTER_API_KEY:
    LLM_PROVIDER = "openrouter"
    LLM_API_KEY = _OPENROUTER_API_KEY
    _DEFAULT_LLM_BASE_URL = "https://openrouter.ai/api/v1"
    _DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"
elif _OPENAI_API_KEY:
    LLM_PROVIDER = "openai"
    LLM_API_KEY = _OPENAI_API_KEY
    _DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
    _DEFAULT_LLM_MODEL = "gpt-4o-mini"
else:
    LLM_PROVIDER = None
    LLM_API_KEY = ""
    _DEFAULT_LLM_BASE_URL = "https://openrouter.ai/api/v1"
    _DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", _DEFAULT_LLM_BASE_URL)
LLM_MODEL = os.environ.get("LLM_MODEL", _DEFAULT_LLM_MODEL)

# LLM-Augmented Alignment: boost/penalize borderline alignment rows
LLM_ALIGNMENT = _env_flag("LLM_ALIGNMENT", True)
# LLM-Based Relation Classification: reclassify top GNN co-occurrence pairs
LLM_RELATIONS = _env_flag("LLM_RELATIONS", True)
# RAG-Based Term Typing: LLM re-ranks the candidates of low-confidence terms
RAG_TYPING = _env_flag("RAG_TYPING", True)

# GNN architecture: 'sage', 'gat', or 'rgcn'
GNN_ARCHITECTURE = os.environ.get("GNN_ARCHITECTURE", "sage")
# GNN output dimension: 384 matches SBERT dim, 64 is legacy
GNN_HIDDEN_DIM = int(os.environ.get("GNN_HIDDEN_DIM", "384"))

# Corpus augmentation: add PubMed co-occurrences between known terms to the graph
CORPUS_AUGMENT = _env_flag("CORPUS_AUGMENT", True)
CORPUS_AUGMENT_PATH = os.environ.get(
    "CORPUS_AUGMENT_PATH", "data/raw/dataset/pubmed_sentences.txt"
)

# GNN co-occurrence relation threshold
GNN_COOCCURRENCE_THRESHOLD = float(os.environ.get("GNN_COOCCURRENCE_THRESHOLD", "0.90"))
GNN_MIN_COOCCURRENCE_COUNT = int(os.environ.get("GNN_MIN_COOCCURRENCE_COUNT", "3"))

# Available domain-specific embedding models (tested and working):
#   'all-MiniLM-L6-v2'                     - General-purpose, 384-dim (default)
#   'allenai/specter'                       - Scientific paper embeddings, 768-dim
#   'allenai/scibert_scivocab_uncased'      - SciBERT with mean pooling, 768-dim
#   'sentence-transformers/all-mpnet-base-v2' - Stronger general model, 768-dim

# Ablation presets: environment overrides plus the first stage that has to be
# re-run (earlier stages are reused from the base run). See run_ablation.py.
ABLATION_PRESETS = {
    'full_pipeline': {
        'description': 'All components enabled (default)',
        'env': {},
        'from_stage': 'builder.py',
    },
    'no_gnn': {
        'description': 'SBERT embeddings only: no GNN in clustering, taxonomy, '
                       'typing or relations',
        'env': {'USE_GNN_EMBEDDINGS': 'false'},
        'from_stage': 'cluster.py',
    },
    'no_bidirectional': {
        'description': 'Forward-only alignment rows (typing is unaffected)',
        'env': {'USE_BIDIRECTIONAL': 'false'},
        'from_stage': 'align.py',
    },
    'embedding_only': {
        'description': 'Pure embedding similarity, no combined lexical scoring',
        'env': {'USE_COMBINED_SCORING': 'false'},
        'from_stage': 'align.py',
    },
    'no_hearst': {
        'description': 'Distributional taxonomy only, no Hearst patterns',
        'env': {'USE_HEARST_PATTERNS': 'false'},
        'from_stage': 'taxonomy.py',
    },
    'no_dep_parsing': {
        'description': 'Pattern-only relation extraction',
        'env': {'USE_DEP_PARSING': 'false'},
        'from_stage': 'relations.py',
    },
    'no_corpus_augment': {
        'description': 'Co-occurrence graph from the annotated papers only',
        'env': {'CORPUS_AUGMENT': 'false'},
        'from_stage': 'builder.py',
    },
    'baseline_sbert': {
        'description': 'Baseline: SBERT only, forward matching, embedding '
                       'similarity, no Hearst/dep-parse/augmentation',
        'env': {'USE_GNN_EMBEDDINGS': 'false', 'USE_BIDIRECTIONAL': 'false',
                'USE_COMBINED_SCORING': 'false', 'USE_HEARST_PATTERNS': 'false',
                'USE_DEP_PARSING': 'false', 'CORPUS_AUGMENT': 'false'},
        'from_stage': 'builder.py',
    },
    'oracle': {
        'description': 'UPPER BOUND: gold NER types allowed (type-match boost, '
                       'NER-type clustering)',
        'env': {'ORACLE_TYPES': 'true'},
        'from_stage': 'cluster.py',
    },
    'domain_specter': {
        'description': 'Full pipeline with SPECTER scientific embeddings',
        'env': {'EMBEDDER_MODEL': 'allenai/specter'},
        'from_stage': 'builder.py',
    },
    'domain_scibert': {
        'description': 'Full pipeline with SciBERT embeddings',
        'env': {'EMBEDDER_MODEL': 'allenai/scibert_scivocab_uncased'},
        'from_stage': 'builder.py',
    },
}


def get_preset(name):
    """Get an ablation preset by name."""
    if name not in ABLATION_PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. "
            f"Available: {list(ABLATION_PRESETS.keys())}"
        )
    return ABLATION_PRESETS[name]


def list_presets():
    """List all available ablation presets."""
    for name, preset in ABLATION_PRESETS.items():
        overrides = ', '.join(f"{k}={v}" for k, v in preset['env'].items())
        print(f"  {name}: {preset['description']}")
        print(f"    env: {overrides or '(defaults)'}; re-run from {preset['from_stage']}")
