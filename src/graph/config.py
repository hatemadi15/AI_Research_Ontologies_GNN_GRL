"""
config.py - Ablation Configuration for Ontology Learning Pipeline

Defines ablation presets that toggle pipeline components:
  - GNN embeddings (vs SBERT-only)
  - Bidirectional alignment (vs forward-only)
  - Combined scoring (vs embedding-only)
  - Hearst patterns (vs distributional-only taxonomy)
  - Dependency parsing (vs pattern-only relations)
  - NER-type clustering (vs embedding-only clustering)
  - LLM validation mode (OpenRouter or OpenAI, see the LLM_* settings)
  - GNN architecture selection (sage/gat/rgcn)
  - Configurable embedding model (domain-specific vs general)
  - Corpus augmentation (PubMed sentences)
"""

import os

# Default embedding model. Override per-preset or via environment variable.
DEFAULT_EMBEDDER_MODEL = 'all-MiniLM-L6-v2'

# LLM validation mode (requires OPENROUTER_API_KEY or OPENAI_API_KEY env var)
LLM_MODE = False

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

# GNN architecture: 'sage', 'gat', or 'rgcn'
GNN_ARCHITECTURE = 'sage'

# Corpus augmentation: add PubMed sentences to co-occurrence graph
CORPUS_AUGMENT = os.environ.get("CORPUS_AUGMENT", "true").lower() == "true"
CORPUS_AUGMENT_PATH = os.environ.get(
    "CORPUS_AUGMENT_PATH", "data/raw/dataset/pubmed_sentences.txt"
)

# LLM-Augmented Alignment: boost/penalize borderline similarity pairs via the configured LLM
LLM_ALIGNMENT = os.environ.get("LLM_ALIGNMENT", "true").lower() == "true"

# LLM-Based Relation Classification: reclassify top GNN co-occurrence pairs
LLM_RELATIONS = os.environ.get("LLM_RELATIONS", "true").lower() == "true"

# RAG-Based Term Typing: use LLM + context to type ambiguous entities
RAG_TYPING = os.environ.get("RAG_TYPING", "true").lower() == "true"

# NER-type-guided clustering (groups entities by NER type first, then sub-clusters)
NER_TYPE_CLUSTERING = os.environ.get("NER_TYPE_CLUSTERING", "true").lower() == "true"

# GNN hidden dimension: 384 matches SBERT dim, 64 is legacy
GNN_HIDDEN_DIM = int(os.environ.get("GNN_HIDDEN_DIM", "384"))

# GNN co-occurrence relation threshold
GNN_COOCCURRENCE_THRESHOLD = float(os.environ.get("GNN_COOCCURRENCE_THRESHOLD", "0.90"))
GNN_MIN_COOCCURRENCE_COUNT = int(os.environ.get("GNN_MIN_COOCCURRENCE_COUNT", "3"))

# Available domain-specific embedding models (tested and working):
#   'all-MiniLM-L6-v2'                     - General-purpose, 384-dim (default)
#   'allenai/specter'                       - Scientific paper embeddings, 768-dim
#   'allenai/scibert_scivocab_uncased'      - SciBERT with mean pooling, 768-dim
#   'sentence-transformers/all-mpnet-base-v2' - Stronger general model, 768-dim

ABLATION_PRESETS = {
    'full_pipeline': {
        'description': 'All components enabled (default)',
        'USE_GNN_EMBEDDINGS': True,
        'USE_BIDIRECTIONAL': True,
        'USE_COMBINED_SCORING': True,
        'USE_HEARST_PATTERNS': True,
        'USE_DEP_PARSING': True,
        'USE_NER_TYPE_CLUSTERING': True,
        'embedder_model': DEFAULT_EMBEDDER_MODEL,
    },
    'no_gnn': {
        'description': 'SBERT embeddings only, no GNN',
        'USE_GNN_EMBEDDINGS': False,
        'USE_BIDIRECTIONAL': True,
        'USE_COMBINED_SCORING': True,
        'USE_HEARST_PATTERNS': True,
        'USE_DEP_PARSING': True,
        'USE_NER_TYPE_CLUSTERING': True,
        'embedder_model': DEFAULT_EMBEDDER_MODEL,
    },
    'no_bidirectional': {
        'description': 'Forward-only alignment (no backward pass)',
        'USE_GNN_EMBEDDINGS': True,
        'USE_BIDIRECTIONAL': False,
        'USE_COMBINED_SCORING': True,
        'USE_HEARST_PATTERNS': True,
        'USE_DEP_PARSING': True,
        'USE_NER_TYPE_CLUSTERING': True,
        'embedder_model': DEFAULT_EMBEDDER_MODEL,
    },
    'embedding_only': {
        'description': 'Pure embedding similarity, no combined scoring',
        'USE_GNN_EMBEDDINGS': True,
        'USE_BIDIRECTIONAL': True,
        'USE_COMBINED_SCORING': False,
        'USE_HEARST_PATTERNS': True,
        'USE_DEP_PARSING': True,
        'USE_NER_TYPE_CLUSTERING': True,
        'embedder_model': DEFAULT_EMBEDDER_MODEL,
    },
    'baseline_sbert': {
        'description': 'Baseline: SBERT only, forward matching, embedding similarity',
        'USE_GNN_EMBEDDINGS': False,
        'USE_BIDIRECTIONAL': False,
        'USE_COMBINED_SCORING': False,
        'USE_HEARST_PATTERNS': False,
        'USE_DEP_PARSING': False,
        'USE_NER_TYPE_CLUSTERING': False,
        'embedder_model': DEFAULT_EMBEDDER_MODEL,
    },
    'no_hearst': {
        'description': 'Distributional taxonomy only, no Hearst patterns',
        'USE_GNN_EMBEDDINGS': True,
        'USE_BIDIRECTIONAL': True,
        'USE_COMBINED_SCORING': True,
        'USE_HEARST_PATTERNS': False,
        'USE_DEP_PARSING': True,
        'USE_NER_TYPE_CLUSTERING': True,
        'embedder_model': DEFAULT_EMBEDDER_MODEL,
    },
    'domain_specter': {
        'description': 'Full pipeline with SPECTER scientific embeddings',
        'USE_GNN_EMBEDDINGS': True,
        'USE_BIDIRECTIONAL': True,
        'USE_COMBINED_SCORING': True,
        'USE_HEARST_PATTERNS': True,
        'USE_DEP_PARSING': True,
        'USE_NER_TYPE_CLUSTERING': True,
        'embedder_model': 'allenai/specter',
    },
    'domain_scibert': {
        'description': 'Full pipeline with SciBERT embeddings',
        'USE_GNN_EMBEDDINGS': True,
        'USE_BIDIRECTIONAL': True,
        'USE_COMBINED_SCORING': True,
        'USE_HEARST_PATTERNS': True,
        'USE_DEP_PARSING': True,
        'USE_NER_TYPE_CLUSTERING': True,
        'embedder_model': 'allenai/scibert_scivocab_uncased',
    },
}


def get_config():
    """Get the default pipeline configuration as a dict.

    Returns a dict with at least 'embedder_model' and all ablation flags
    from the full_pipeline preset.
    """
    conf = dict(ABLATION_PRESETS['full_pipeline'])
    conf['embedder_model'] = conf.get('embedder_model', DEFAULT_EMBEDDER_MODEL)
    return conf


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
    for name, config in ABLATION_PRESETS.items():
        print(f"  {name}: {config['description']}")
        flags = {k: v for k, v in config.items()
                 if k.startswith('USE_')}
        disabled = [k for k, v in flags.items() if not v]
        if disabled:
            print(f"    Disabled: {', '.join(disabled)}")
        else:
            print(f"    All flags enabled")
