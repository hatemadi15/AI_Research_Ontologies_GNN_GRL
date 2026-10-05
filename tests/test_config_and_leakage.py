"""LLM provider selection, RAG answer parsing and a static gold-leakage check."""

import importlib
import os
import re

import pytest

import config
import rag_typing

SRC_DIR = os.path.dirname(os.path.abspath(config.__file__))


@pytest.fixture
def reload_config(monkeypatch):
    def _reload(**env):
        for key in ('OPENROUTER_API_KEY', 'OPENAI_API_KEY', 'LLM_BASE_URL', 'LLM_MODEL'):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(config)
    yield _reload
    importlib.reload(config)


def test_openrouter_key_uses_openrouter(reload_config):
    cfg = reload_config(OPENROUTER_API_KEY='or-key')
    assert cfg.LLM_PROVIDER == 'openrouter'
    assert cfg.LLM_BASE_URL == 'https://openrouter.ai/api/v1'
    assert cfg.LLM_MODEL == 'openai/gpt-4o-mini'


def test_openai_key_is_never_sent_to_openrouter(reload_config):
    cfg = reload_config(OPENAI_API_KEY='oa-key')
    assert cfg.LLM_PROVIDER == 'openai'
    assert cfg.LLM_BASE_URL == 'https://api.openai.com/v1'
    assert cfg.LLM_MODEL == 'gpt-4o-mini'
    assert cfg.LLM_API_KEY == 'oa-key'


def test_overrides_and_no_key(reload_config):
    cfg = reload_config(OPENAI_API_KEY='oa-key', LLM_MODEL='gpt-x',
                        LLM_BASE_URL='https://example.test/v1')
    assert (cfg.LLM_BASE_URL, cfg.LLM_MODEL) == ('https://example.test/v1', 'gpt-x')
    assert reload_config().LLM_API_KEY == ''


def test_oracle_flags_default_off(reload_config, monkeypatch):
    monkeypatch.delenv('ORACLE_TYPES', raising=False)
    monkeypatch.delenv('NER_TYPE_CLUSTERING', raising=False)
    cfg = reload_config()
    assert cfg.ORACLE_TYPES is False
    assert cfg.NER_TYPE_CLUSTERING is False
    monkeypatch.setenv('ORACLE_TYPES', 'true')
    cfg = reload_config()
    assert cfg.NER_TYPE_CLUSTERING is True


def test_rag_answer_parsing_prefers_exact_choice():
    candidates = [['u1', 'Fatigue', 0.5], ['u2', 'Fatigue strength', 0.4]]
    assert rag_typing.parse_choice('2', candidates) == 'u2'
    assert rag_typing.parse_choice('2.', candidates) == 'u2'
    assert rag_typing.parse_choice('Fatigue strength', candidates) == 'u2'
    assert rag_typing.parse_choice('NONE', candidates) is None
    assert rag_typing.parse_choice('7', candidates) is None
    assert rag_typing.parse_choice('fatigue strength is best', candidates) is None


def test_gold_types_only_read_by_oracle_or_evaluation_code():
    """Gold labels may only reach predictions behind ORACLE_TYPES.

    Every module that refers to the gold types file must be the writer
    (builder.py), the definition (config.py) or gate its use on ORACLE_TYPES /
    NER_TYPE_CLUSTERING. The old file name must not be used anywhere.
    """
    allowed_without_gate = {'builder.py', 'config.py'}
    for name in sorted(os.listdir(SRC_DIR)):
        if not name.endswith('.py'):
            continue
        with open(os.path.join(SRC_DIR, name), encoding='utf-8') as f:
            source = f.read()
        assert 'entity_types.json' not in source, name
        if re.search(r'GOLD_TYPES_FILE|gold_term_types', source) and \
                name not in allowed_without_gate:
            assert re.search(r'ORACLE_TYPES|NER_TYPE_CLUSTERING', source), name
