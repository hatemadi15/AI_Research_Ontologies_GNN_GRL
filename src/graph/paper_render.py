"""
paper_render.py - render paper A's Markdown sources for submission

The sources (docs/PAPER_A.md, docs/PAPER_A_SI.md) cite by bibliography key, e.g.
"(dai2020analysis; torres2024augmentation)". This script numbers the cited works in the order of
their first citation in each file, replaces the keys by bracketed numbers ("[5,6]"; a run of three
or more becomes a range, "[5–7]") and writes the reference list from docs/references.bib in the
journal's style (authors as "Surname, I.", up to five, else the first and "et al."; title; venue;
volume, pages or article number; year; DOI). The main text's "## References" section is filled;
the SI gets a "## Supplementary references" section at its end. The main text's tables, each
marked in the source by "<!-- table -->" and "<!-- /table -->" with its caption and note, move to
a "## Tables" section at the end, as npj Computational Materials asks; references are numbered
in the journal's order, the text and Methods first, then the tables, then the figure legends.
Nothing else changes.

Usage:
  python src/graph/paper_render.py [--bib docs/references.bib] [--out docs/submission]
      docs/PAPER_A.md docs/PAPER_A_SI.md
"""

import argparse
import os
import re
import unicodedata

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
KEY = r'[a-z]+\d{4}[a-z]+'
TABLE = re.compile(r'<!-- table -->\n+(.*?)\n+<!-- /table -->\n?', re.S)


def parse_bib(text):
    """{key: {'type': entry type, field: value}} from a BibTeX string with braced fields."""
    entries = {}
    for m in re.finditer(r'@(\w+)\s*\{\s*([^,\s]+)\s*,', text):
        etype, key, i = m.group(1).lower(), m.group(2), m.end()
        fields, depth, start = {}, 1, i
        # the entry runs to the brace that closes the opening one
        while i < len(text) and depth:
            depth += {'{': 1, '}': -1}.get(text[i], 0)
            i += 1
        body = text[start:i - 1]
        for f in re.finditer(r'(\w+)\s*=\s*', body):
            j = f.end()
            if j < len(body) and body[j] == '{':
                d, k = 0, j
                while k < len(body):
                    d += {'{': 1, '}': -1}.get(body[k], 0)
                    k += 1
                    if d == 0:
                        break
                fields[f.group(1).lower()] = body[j + 1:k - 1]
            else:
                value = re.match(r'[^,\n]*', body[j:]).group(0).strip()
                fields.setdefault(f.group(1).lower(), value)
        entries[key] = {'type': etype, **fields}
    return entries


ACCENTS = {'"': '\u0308', "'": '\u0301', '`': '\u0300', '^': '\u0302', '~': '\u0303'}
LETTERS = {'o': 'ø', 'O': 'Ø', 'ss': 'ß', 'ae': 'æ', 'aa': 'å', 'l': 'ł', 'i': 'ı'}


def unlatex(s):
    """LaTeX accents and letters ({\\"u}, \\'e, {\\o}, \\'{\\i}) as Unicode."""
    s = re.sub(r'\\([&%_$#])', r'\1', s)
    s = re.sub(r'\\([a-zA-Z]+)\b\s*', lambda m: LETTERS.get(m.group(1), m.group(0)), s)
    s = re.sub(r'\\(["\'`^~])\s*\{?\s*([A-Za-zı])\s*\}?',
               lambda m: unicodedata.normalize('NFC', m.group(2).replace('ı', 'i') + ACCENTS[m.group(1)]), s)
    return s


def plain(s):
    """A field's text in Unicode, without braces, its whitespace collapsed and '--' as an en dash."""
    s = unlatex(s)
    return re.sub(r'\s+', ' ', s.replace('{', '').replace('}', '')).replace('--', '–').strip()


def split_authors(field):
    """The names of an author field, split on ' and ' outside braces."""
    names, depth, cur = [], 0, ''
    tokens = re.split(r'(\{|\}|\s+and\s+)', field)
    for t in tokens:
        if t == '{':
            depth += 1
        elif t == '}':
            depth -= 1
        if depth == 0 and re.fullmatch(r'\s+and\s+', t or ''):
            names.append(cur)
            cur = ''
        else:
            cur += t
    names.append(cur)
    return [n.strip() for n in names if n.strip()]


def initials(first):
    """'Ming-Wei' -> 'M.-W.'; 'N. M. Anoop' -> 'N. M. A.'."""
    parts = []
    for word in first.split():
        parts.append('-'.join(p[0] + '.' for p in word.split('-') if p))
    return ' '.join(parts)


def short_name(name):
    """'Surname, I.' of a BibTeX name ('Last, First', 'First Last' or a braced single name)."""
    if name.startswith('{') and name.endswith('}'):
        return plain(name)
    name = plain(name)
    if ',' in name:
        last, first = [x.strip() for x in name.split(',', 1)]
    else:
        *first_parts, last = name.split()
        first = ' '.join(first_parts)
    return f'{last}, {initials(first)}' if first else last


def format_authors(field):
    names = split_authors(field)
    if not names:
        return ''
    if len(names) > 5 or names[-1].strip() == 'others':
        return short_name(names[0]) + ' et al.'
    short = [short_name(n) for n in names]
    return short[0] if len(short) == 1 else ', '.join(short[:-1]) + ' & ' + short[-1]


def format_entry(e):
    """One reference in the journal's style."""
    authors = format_authors(e.get('author', ''))
    title = plain(e.get('title', '')).rstrip('.')
    year = plain(e.get('year', ''))
    pages = plain(e.get('pages', ''))
    if e['type'] == 'article':
        venue = f"*{plain(e.get('journal', ''))}*"
        vol = plain(e.get('volume', ''))
        where = venue + (f' **{vol}**' if vol else '') + (f', {pages}' if pages else '')
        tail = f'{where} ({year}).'
    elif e['type'] == 'inproceedings':
        venue = f"In *{plain(e.get('booktitle', ''))}*"
        pub = plain(e.get('publisher', ''))
        tail = venue + (f' {pages}' if pages else '') + f" ({pub + ', ' if pub else ''}{year})."
    else:
        tail = f'({year}).'
    if authors and not authors.endswith('.'):
        authors += '.'
    out = f'{authors} {title}. {tail}' if authors else f'{title}. {tail}'
    if e.get('eprint') and plain(e.get('archiveprefix', '')).lower() == 'arxiv':
        out += f" Preprint at arXiv:{plain(e['eprint'])}."
    if e.get('doi'):
        out += f" https://doi.org/{plain(e['doi'])}"
    elif e.get('url') and e['type'] == 'misc':
        out += f" {plain(e['url'])}"
    return out


def cited_keys(text, keys):
    """The bibliography keys of `text` in the order of their first occurrence."""
    seen = []
    for m in re.finditer(r'\b(' + KEY + r')\b', text):
        if m.group(1) in keys and m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def compress(numbers):
    """'5,6' or '5–7,9' for a sorted list of numbers (a run of three or more is a range)."""
    numbers, out, i = sorted(set(numbers)), [], 0
    while i < len(numbers):
        j = i
        while j + 1 < len(numbers) and numbers[j + 1] == numbers[j] + 1:
            j += 1
        out.append(f'{numbers[i]}–{numbers[j]}' if j - i >= 2 else
                   ','.join(str(n) for n in numbers[i:j + 1]))
        i = j + 1
    return ','.join(out)


def number_citations(text, number):
    """`text` with each cited key replaced by its number: a parenthesis of keys alone becomes one
    bracket, any other key a bracket of its own."""
    def group(m):
        keys = re.split(r';\s*', m.group(1))
        if not all(k in number for k in keys):
            return m.group(0)
        return '[' + compress([number[k] for k in keys]) + ']'
    text = re.sub(r'\((' + KEY + r'(?:;\s*' + KEY + r')*)\)', group, text)
    return re.sub(r'\b(' + KEY + r')\b',
                  lambda m: f'[{number[m.group(1)]}]' if m.group(1) in number else m.group(0), text)


def split_tables(text):
    """The text without its marked table blocks, and the blocks (caption, table and note) in
    the order of their table numbers; the blank lines left around a removed block collapse to
    one."""
    blocks = [m.group(1).strip('\n') for m in TABLE.finditer(text)]
    order = [int(m.group(1)) if m else 0
             for m in (re.match(r'\*\*Table (\d+)\.', b) for b in blocks)]
    blocks = [b for _, b in sorted(zip(order, blocks), key=lambda x: x[0])]
    return re.sub(r'\n{3,}', '\n\n', TABLE.sub('', text)), blocks


def render(text, bib, heading, move_tables=False):
    """The rendered text: numbered citations, and the reference list under `heading`, which
    replaces that section's body if the text has it and is appended otherwise. With
    move_tables the marked tables go to a "## Tables" section at the end, and the numbering
    follows the text before the reference list, then the tables, then the rest; otherwise
    the markers are dropped and the tables stay in place."""
    if move_tables:
        text, tables = split_tables(text)
    else:
        text, tables = TABLE.sub(lambda m: m.group(1).strip('\n') + '\n', text), []
    m = re.search(r'(?m)^## ' + re.escape(heading) + r'$', text)
    before, after = (text[:m.start()], text[m.start():]) if m else (text, '')
    keys = cited_keys(before + '\n'.join(tables) + after, bib)
    number = {k: i for i, k in enumerate(keys, 1)}
    body = number_citations(text, number)
    listing = '\n'.join(f'{number[k]}. {format_entry(bib[k])}' for k in keys)
    section = f'## {heading}\n\n{listing}\n'
    m = re.search(r'(?m)^## ' + re.escape(heading) + r'\n.*?(?=^## |\Z)', body, re.S)
    if m:
        body = body[:m.start()] + section + ('\n' if m.end() < len(body) else '') + body[m.end():]
    else:
        body = body.rstrip('\n') + '\n\n' + section
    if tables:
        body = (body.rstrip('\n') + '\n\n## Tables\n\n'
                + '\n\n'.join(number_citations(t, number) for t in tables) + '\n')
    return body


def main(argv=None):
    parser = argparse.ArgumentParser(description='Render paper A with numbered references')
    parser.add_argument('sources', nargs='*', default=[os.path.join(ROOT, 'docs', 'PAPER_A.md'),
                                                       os.path.join(ROOT, 'docs', 'PAPER_A_SI.md')])
    parser.add_argument('--bib', default=os.path.join(ROOT, 'docs', 'references.bib'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'docs', 'submission'))
    args = parser.parse_args(argv)
    with open(args.bib) as f:
        bib = parse_bib(f.read())
    os.makedirs(args.out, exist_ok=True)
    for path in args.sources:
        with open(path) as f:
            text = f.read()
        main_text = bool(re.search(r'(?m)^## References$', text))
        heading = 'References' if main_text else 'Supplementary references'
        out = render(text, bib, heading, move_tables=main_text)
        target = os.path.join(args.out, os.path.basename(path))
        with open(target, 'w') as f:
            f.write(out)
        print(f'wrote {target} ({len(cited_keys(text, bib))} references)')


if __name__ == '__main__':
    main()
