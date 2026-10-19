"""The word tokenizer of ChemDataExtractor 1.3.0 (chem_tokenize.py), which cut SOFC-Exp's training
and development articles into tokens.

The SOFC-Exp files hold tokens, not the article texts; joining a sentence's tokens with spaces
and tokenizing again gives the same tokens on every sentence of the training and development
articles. (On the release's texts, with their offsets, the tokenizer gives the corpus's tokens
on all 7,630 sentences of those 34 articles; ROADMAP §4, week-6 amendment.) The 11 test
articles were tokenized otherwise, with Java's BreakIterator.
"""
import json
import os
from collections import Counter

from chem_tokenize import ChemWordTokenizer
from config import RAW_DIR

SOFC = os.path.join(RAW_DIR, 'sofc_exp')


def test_chemical_formulas_units_and_hyphens():
    tok = ChemWordTokenizer()
    assert tok.tokenize('The Ba0.5Sr0.5Co0.8Fe0.2O3−δ cathode on a NiO-YSZ anode reached 0.5 W '
                        'cm−2 at 800 °C (Fig. 3).') == [
        'The', 'Ba0.5Sr0.5Co0.8Fe0.2O3−δ', 'cathode', 'on', 'a', 'NiO-YSZ', 'anode', 'reached',
        '0.5', 'W', 'cm−2', 'at', '800', '°', 'C', '(', 'Fig.', '3', ')', '.']
    assert tok.tokenize('Low-temperature SOFCs with a gadolinium-doped ceria (GDC) layer were '
                        'tested for 100 h.') == [
        'Low', '-', 'temperature', 'SOFCs', 'with', 'a', 'gadolinium', '-', 'doped', 'ceria',
        '(', 'GDC', ')', 'layer', 'were', 'tested', 'for', '100', 'h', '.']
    text = 'The cell reached 0.5 W cm−2 at 800 °C with Ni-YSZ.'
    assert [text[a:b] for a, b in tok.span_tokenize(text)] == tok.tokenize(text)


def test_it_gives_back_the_tokens_of_sofc_exps_training_and_development_articles():
    with open(os.path.join(SOFC, 'context.json'), encoding='utf-8') as f:
        context = json.load(f)
    with open(os.path.join(SOFC, 'split.json'), encoding='utf-8') as f:
        split = json.load(f)
    tok = ChemWordTokenizer()
    same = Counter()
    for doc, part in split.items():
        for words in context[doc]['sentences']:
            same[part, tok.tokenize(' '.join(words)) == words] += 1
    assert same['train', True] == 6026 and same['dev', True] == 1604
    assert not same['train', False] and not same['dev', False]
    # the test articles' tokens are Java BreakIterator's: 618 of 1,836 sentences come out
    # otherwise (Fig ., cm − 2, O2 − ... are split where ChemDataExtractor keeps them)
    assert (same['test', True], same['test', False]) == (1218, 618)
