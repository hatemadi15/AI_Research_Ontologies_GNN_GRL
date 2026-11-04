"""
build_tacl.py - typeset paper B's TACL draft in TACL's style and count its pages

Reads docs/PAPER_B_TACL.md and docs/references.bib; writes docs/submission/PAPER_B_TACL.tex
and, where pdflatex, bibtex and latexmk are installed, docs/submission/PAPER_B_TACL.pdf, with
the pages of the main text, the references and each appendix. Nothing in the draft's wording
changes; the conversion only maps its conventions onto LaTeX:

- the first heading is the title and `## Abstract` the abstract; the line "*Anonymous TACL
  submission*" is dropped, since tacl2021v1.sty prints it in its submission mode;
- each citation key of the reference list becomes \\citealp{key} where it stands inside
  parentheses ("(benjamini1995controlling)" reads "(Benjamini and Hochberg, 1995)") and
  \\citet{key} in running text ("Following augenstein2017generalisation" reads "Following
  Augenstein et al. (2017)");
- `## References` becomes acl_natbib's reference list over docs/references.bib, with the
  companion paper's entry ("Anonymous. 2026. Title withheld for review. Under review.") from
  COMPANION, cited where the draft writes "Anonymous, 2026";
- the headings lose their written numbers, which LaTeX gives again in the same order (checked:
  `check_numbering`), "Use of an AI assistant" is unnumbered, and "Appendix A: ..." opens
  \\appendix;
- tables and figures become floats with their written captions (docs/tacl/tables.lua);
- the one formula in Unicode (A.5) is set as math (FORMULAS).

The style files in docs/tacl are tacl2021v1.sty (2021/12/15) and acl_natbib.bst, the files of
TACL's author kit; STYLE_SHA256 pins them. The PDF's properties carry the title and no author.

Usage:
  python src/graph/build_tacl.py [--source docs/PAPER_B_TACL.md] [--tex-only]
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'docs' / 'PAPER_B_TACL.md'
STYLE = ROOT / 'docs' / 'tacl'
BIB = ROOT / 'docs' / 'references.bib'
FIGURES = ROOT / 'docs' / 'figures'
OUT = ROOT / 'docs' / 'submission'
NAME = 'PAPER_B_TACL'
# the author kit's files (two arXiv source packages that use them carry the same bytes)
STYLE_SHA256 = {
    'tacl2021v1.sty': '7c71ede112ccb3711c7073c4b1c60abfb076b83945e0e9d14ed79caa37dca3b6',
    'acl_natbib.bst': '27a3b62fdc4211fa56fe4827572ff0524b3ba3fe96c9443faaba5a683af83c08',
}
COMPANION_KEY = 'anonymous2026companion'
COMPANION = ('@misc{' + COMPANION_KEY + ',\n  author = {Anonymous},\n'
             '  title = {Title withheld for review},\n  year = {2026},\n'
             '  note = {Under review}\n}\n')
COMPANION_LINE = '- Anonymous. 2026. Title withheld for review. Under review.'
FORMULAS = {'S²/(1 − ρ̄)·(n + Σρ)/n²': r'$S^2/(1-\bar{\rho})\cdot(n+\Sigma\rho)/n^2$'}
DROPPED = '*Anonymous TACL submission*'
UNNUMBERED = ('Use of an AI assistant',)


class BuildError(RuntimeError):
    """The draft does not have the shape the conversion expects; the message says where."""


def sections(text):
    """[(level, heading, start, end)] of the markdown headings, with the span of each heading's
    body (to the next heading of any level)."""
    heads = [(len(m.group(1)), m.group(2).strip(), m.start(), m.end())
             for m in re.finditer(r'^(#{1,6}) (.+)$', text, re.M)]
    return [(level, title, start, heads[i + 1][2] if i + 1 < len(heads) else len(text))
            for i, (level, title, start, _) in enumerate(heads)]


def split_draft(text):
    """(title, abstract, main, references, appendix) of the draft: the parts between its
    `# title`, `## Abstract`, `## References` and `## Appendix A` headings."""
    first = text.split('\n', 1)[0]
    if not first.startswith('# '):
        raise BuildError('the draft does not open with its title heading')
    title = first[2:].strip()
    marks = {}
    for heading in ('## Abstract', '## References', '## Appendix A'):
        found = [m.start() for m in re.finditer('^' + re.escape(heading), text, re.M)]
        if len(found) != 1:
            raise BuildError(f'{heading!r} occurs {len(found)} times')
        marks[heading] = found[0]
    a, r, x = marks['## Abstract'], marks['## References'], marks['## Appendix A']
    if not a < r < x:
        raise BuildError('the abstract, references and appendix are out of order')
    after_abstract = re.search(r'^## ', text[a + 3:], re.M)
    abstract_end = a + 3 + after_abstract.start()
    abstract = text[a:abstract_end].split('\n', 1)[1].strip()
    return title, abstract, text[abstract_end:r], text[r:x], text[x:]


def reference_keys(references):
    """The citation keys of the reference list, in its order; the companion paper's line must be
    there too."""
    keys = re.findall(r'^- `([^`]+)`', references, re.M)
    if COMPANION_LINE not in references:
        raise BuildError("the reference list has no companion entry ('Anonymous. 2026. ...')")
    return keys


def paragraph_depth(text, position):
    """How many parentheses are open at `position` within its paragraph."""
    start = text.rfind('\n\n', 0, position)
    chunk = text[start + 2 if start >= 0 else 0:position]
    return chunk.count('(') - chunk.count(')')


def cite(text, keys):
    """Every bare occurrence of a key as \\citealp{key} inside parentheses, else \\citet{key};
    a key inside backticks or already in a command is left alone."""
    if not keys:
        return text
    pattern = re.compile(r'(?<![`{\w])(' + '|'.join(map(re.escape, sorted(keys, key=len,
                                                                         reverse=True)))
                         + r')(?![`}\w])')

    def one(m):
        command = 'citealp' if paragraph_depth(text, m.start()) > 0 else 'citet'
        return f'\\{command}{{{m.group(1)}}}'
    return pattern.sub(one, text)


def numbered(heading):
    """(the written number, the heading without it) of '3.1 Our runs ...', 'A.5 The ...' or
    'Appendix A: Replication material'; (None, heading) when it carries none."""
    m = re.match(r'^Appendix ([A-Z]): (.+)$', heading)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r'^((?:\d+|[A-Z])(?:\.\d+)*) (.+)$', heading)
    if m:
        return m.group(1), m.group(2)
    return None, heading


def check_numbering(main, appendix):
    """The written section numbers are the ones LaTeX gives the same headings: 1, 2, ... with
    subsections 3.1, 3.2, ... in the main text, and A, B with A.1, ... after \\appendix; each
    discrepancy is named. Unnumbered headings (UNNUMBERED) are skipped."""
    wrong = []
    for part, letters in ((main, False), (appendix, True)):
        section, sub = 0, 0
        for level, heading, _, _ in sections(part):
            number, rest = numbered(heading)
            if rest in UNNUMBERED or level not in (2, 3):
                continue
            if level == 2:
                section, sub = section + 1, 0
                expected = chr(64 + section) if letters else str(section)
            else:
                sub += 1
                expected = f'{chr(64 + section) if letters else section}.{sub}'
            if number != expected:
                wrong.append(f'{heading!r}: written {number}, LaTeX gives {expected}')
    if wrong:
        raise BuildError('the written numbers differ from LaTeX\'s:\n  ' + '\n  '.join(wrong))


def strip_numbers(part):
    """The part's headings without their written numbers; UNNUMBERED headings marked {-}."""
    def one(m):
        _, rest = numbered(m.group(2).strip())
        mark = ' {-}' if rest in UNNUMBERED else ''
        return f'{m.group(1)} {rest}{mark}'
    return re.sub(r'^(#{2,3}) (.+)$', one, part, flags=re.M)


def formulas(text):
    for plain, math in FORMULAS.items():
        if plain not in text:
            raise BuildError(f'the formula {plain!r} is not in the draft')
        text = text.replace(plain, math)
    return text


def preprocess(text):
    """(title, abstract, body) in pandoc markdown, ready for docs/tacl/template.tex; the body
    holds the reference list and \\appendix as raw LaTeX."""
    if DROPPED not in text:
        raise BuildError(f'{DROPPED!r} is not in the draft')
    text = text.replace(DROPPED + '\n', '', 1)
    title, abstract, main, references, appendix = split_draft(text)
    keys = reference_keys(references)
    check_numbering(main, appendix)
    # a named destination where the main text ends, for page_counts (no typeset output)
    raw = ('```{=latex}\n\\pdfdest name{' + MARKS['references'] + '} xyz\\relax\n'
           '\\nocite{' + COMPANION_KEY + '}\n\\bibliographystyle{acl_natbib}\n'
           '\\bibliography{references,companion}\n\\appendix\n```\n\n')
    body = strip_numbers(main) + raw + strip_numbers(appendix)
    body = formulas(cite(body, keys))
    return title, cite(abstract, keys), body, keys


def check_style(directory=STYLE):
    """The style files are the author kit's (STYLE_SHA256)."""
    for name, digest in STYLE_SHA256.items():
        found = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if found != digest:
            raise BuildError(f'{name} is not the author kit\'s file (sha256 {found})')


def to_latex(title, abstract, body, workdir):
    """docs/tacl/template.tex filled by pandoc (with tables.lua) from the preprocessed parts."""
    md = workdir / 'body.md'
    md.write_text(body)
    abstract_tex = subprocess.run(
        ['pandoc', '-f', 'markdown', '-t', 'latex', '--wrap=preserve'], input=abstract,
        capture_output=True, text=True, check=True).stdout.strip()
    out = subprocess.run(
        ['pandoc', str(md), '-f', 'markdown', '-t', 'latex', '--wrap=preserve',
         '--shift-heading-level-by=-1', '--lua-filter', str(STYLE / 'tables.lua'),
         '--template', str(STYLE / 'template.tex'),
         '--metadata', f'title={title}', '--variable', f'abstract={abstract_tex}'],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise BuildError('pandoc failed:\n' + out.stderr[-2000:])
    return out.stdout


# the named destinations where the main text ends (`preprocess` sets it before the reference
# list) and where the two appendices open (hyperref's, under \appendix)
MARKS = {'references': 'tacl-references', 'appendix_a': 'appendix.A', 'appendix_b': 'appendix.B'}
# the text block of tacl2021v1.sty on A4, in PDF points from the bottom: top margin -0.5 cm,
# head 5 pt, text height 24.7 cm
TEXT_TOP = 841.89 - (72 - 0.5 / 2.54 * 72 + 5)
TEXT_HEIGHT = 24.7 / 2.54 * 72


def destinations(pdf):
    """{name: (page, x, y)} of the PDF's named destinations (pdfinfo -dests)."""
    out = subprocess.run(['pdfinfo', '-dests', str(pdf)], capture_output=True, text=True,
                         check=True).stdout
    return {name: (int(page), float(x), float(y)) for page, x, y, name in re.findall(
        r'^\s*(\d+) \[ XYZ\s+([\d.]+)\s+([\d.]+)\s+\S+\s*\] "([^"]+)"', out, re.M)}


def position(mark, width=595.28):
    """A destination's place in pages of two columns: its page less 1, plus the share of the
    page's two columns before it, read column by column."""
    page, x, y = mark
    column = 0 if x < width / 2 else 1
    within = min(max((TEXT_TOP - y) / TEXT_HEIGHT, 0.0), 1.0)
    return round(page - 1 + (column + within) / 2, 2)


def page_counts(pdf):
    """The pages of the main text (to the references), the references and each appendix,
    in pages of two columns, from the named destinations of MARKS."""
    found = destinations(pdf)
    missing = [m for m in MARKS.values() if m not in found]
    if missing:
        raise BuildError(f'destinations not found in the PDF: {missing}')
    total = int(subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True,
                               check=True).stdout.split('Pages:')[1].split()[0])
    at = {k: position(found[m]) for k, m in MARKS.items()}
    return {'pages': total, 'main_text': at['references'],
            'references': round(at['appendix_a'] - at['references'], 2),
            'appendix_a': round(at['appendix_b'] - at['appendix_a'], 2),
            'appendix_b': round(total - at['appendix_b'], 2),
            'starts': {k: found[m][0] for k, m in MARKS.items()}}


def pdf_properties(pdf):
    """pdfinfo's Title, Author, Creator and Producer lines."""
    info = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True,
                          check=True).stdout
    return {k: v.strip() for k, v in re.findall(r'^(\w+):[ \t]*(.*)$', info, re.M)
            if k in ('Title', 'Author', 'Creator', 'Producer', 'Subject', 'Keywords')}


def shown(path):
    """A path relative to the repository when it is inside it."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def build(source=SOURCE, out=OUT, tex_only=False):
    check_style()
    title, abstract, body, keys = preprocess(Path(source).read_text())
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        tex = to_latex(title, abstract, body, work)
        (out / f'{NAME}.tex').write_text(tex)
        result = {'tex': shown(out / f'{NAME}.tex'), 'citation_keys': len(keys)}
        if tex_only or not all(shutil.which(t) for t in ('pdflatex', 'bibtex', 'latexmk')):
            return dict(result, pdf=None)
        for name in STYLE_SHA256:
            shutil.copy(STYLE / name, work / name)
        shutil.copy(BIB, work / 'references.bib')
        (work / 'companion.bib').write_text(COMPANION)
        os.symlink(FIGURES, work / 'figures')
        (work / f'{NAME}.tex').write_text(tex)
        run = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                              f'{NAME}.tex'], cwd=work, capture_output=True, text=True,
                             errors='replace')
        if run.returncode != 0:
            log = (work / f'{NAME}.log').read_text(errors='replace')
            raise BuildError('latexmk failed:\n' + log[-3000:])
        log = (work / f'{NAME}.log').read_text(errors='replace')
        shutil.copy(work / f'{NAME}.pdf', out / f'{NAME}.pdf')
        warnings = {'undefined_citations': sorted(set(re.findall(
                        r"Citation `([^']+)' .*undefined", log))),
                    'overfull_hboxes': len(re.findall(r'^Overfull \\hbox', log, re.M)),
                    'float_too_large': [f'{by} at line {line}' for by, line in re.findall(
                        r'Float too large for page by ([\d.]+pt) on input line (\d+)', log)]}
    pdf = out / f'{NAME}.pdf'
    return dict(result, pdf=shown(pdf), **page_counts(pdf),
                properties=pdf_properties(pdf), warnings=warnings)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Typeset paper B's TACL draft")
    parser.add_argument('--source', default=str(SOURCE))
    parser.add_argument('--tex-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = build(args.source, tex_only=args.tex_only)
    except BuildError as err:
        print(f'build_tacl: {err}', file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
