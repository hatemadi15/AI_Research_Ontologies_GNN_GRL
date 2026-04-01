"""
config.py - Ablation Configuration for Ontology Learning Pipeline

Defines ablation presets that toggle pipeline components:
  - GNN embeddings (vs SBERT-only)
  - Bidirectional alignment (vs forward-only)
  - Combined scoring (vs embedding-only)
  - Hearst patterns (vs distributional-only taxonomy)
  - Dependency parsing (vs pattern-only relations)
  - NER-type clustering (vs embedding-only clustering)
  - Configurable embedding model (domain-specific vs general)
"""

# Default embedding model. Override per-preset or via environment variable.
DEFAULT_EMBEDDER_MODEL = 'all-MiniLM-L6-v2'

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
