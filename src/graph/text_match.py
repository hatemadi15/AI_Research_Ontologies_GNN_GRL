"""
text_match.py - Deterministic term matching in free text

Shared by builder.py (PubMed co-occurrence), taxonomy.py (Hearst patterns),
relations.py (entity resolution) and rag_typing.py (context retrieval).

The previous code matched terms with raw substring tests (`term in sentence`),
so "al" matched inside "material", and resolved ambiguous matches by iterating
a Python set, whose order depends on the hash seed. TermMatcher instead uses
one compiled regex alternation with word boundaries, longest term first, which
gives deterministic leftmost-longest, non-overlapping matches.
"""

import re

_SPACE_RE = re.compile(r'\s+')
_NON_ALNUM_RE = re.compile(r'[^a-z0-9 ]+')
_DETERMINERS = ('the ', 'a ', 'an ', 'this ', 'these ', 'those ')


def normalize_text(text):
    """Lowercase and collapse whitespace."""
    return _SPACE_RE.sub(' ', str(text).lower()).strip()


def _singular(word):
    if len(word) > 3 and word.endswith('ies'):
        return word[:-3] + 'y'
    if len(word) > 3 and word.endswith('es') and word[-3] in 'sxz':
        return word[:-2]
    if len(word) > 2 and word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word


def lemma_key(term):
    """Crude lemma key used to keep near-duplicate terms together.

    'cracks' -> 'crack', 'the carbon steels' -> 'carbon steel'. Used to group
    surface variants into the same cross-validation fold.
    """
    t = normalize_text(term)
    for det in _DETERMINERS:
        if t.startswith(det):
            t = t[len(det):]
    t = _NON_ALNUM_RE.sub(' ', t.replace('-', ' '))
    words = [_singular(w) for w in t.split()]
    return ' '.join(words)


class TermMatcher:
    """Find known terms in text with word boundaries, longest match first."""

    def __init__(self, terms, min_length=1):
        self.terms = sorted({normalize_text(t) for t in terms
                             if len(normalize_text(t)) >= min_length},
                            key=lambda t: (-len(t), t))
        if self.terms:
            alternation = '|'.join(re.escape(t) for t in self.terms)
            self._pattern = re.compile(rf'(?<!\w)(?:{alternation})(?!\w)')
        else:
            self._pattern = None

    def find(self, text):
        """Return [(start, end, term)] for non-overlapping matches in `text`.

        Matching is case-insensitive (the text is lowercased; offsets refer to
        the lowercased text, which has the same length).
        """
        if self._pattern is None:
            return []
        lowered = str(text).lower()
        return [(m.start(), m.end(), m.group(0))
                for m in self._pattern.finditer(lowered)]

    def terms_in(self, text):
        """Set of distinct known terms occurring in `text`."""
        return {term for _, _, term in self.find(text)}

    def match_phrase(self, phrase):
        """Resolve a free-text phrase to a single known term, or None.

        Exact match wins; otherwise the longest known term contained in the
        phrase (by word boundaries) is returned. This replaces the old fuzzy
        substring matching that iterated an unordered set.
        """
        phrase_norm = normalize_text(phrase)
        for det in _DETERMINERS:
            if phrase_norm.startswith(det):
                phrase_norm = phrase_norm[len(det):]
        if not phrase_norm:
            return None
        matches = self.find(phrase_norm)
        for start, end, term in matches:
            if start == 0 and end == len(phrase_norm):
                return term
        if not matches:
            return None
        return max(matches, key=lambda m: (m[1] - m[0], -m[0]))[2]
