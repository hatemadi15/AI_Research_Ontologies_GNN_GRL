"""The TACL build (build_tacl.py): the citation mapping, the heading numbers, the preprocessing
of the committed draft, the pinned style files, the LaTeX pandoc writes (when pandoc is here)
and the PDF with its page counts (when pdflatex, bibtex, latexmk and pdfinfo are here)."""
import re
import shutil
from pathlib import Path

import pytest

import build_tacl as bt


def test_citations_inside_parentheses_and_in_running_text():
    keys = ['smith2020a', 'lee2019b']
    text = ('Seen names (smith2020a) differ.\n\nAs lee2019b find, (see lee2019b; smith2020a) '
            'and `smith2020a` stay.')
    out = bt.cite(text, keys)
    assert '(\\citealp{smith2020a})' in out
    assert 'As \\citet{lee2019b} find' in out
    assert '(see \\citealp{lee2019b}; \\citealp{smith2020a})' in out
    assert '`smith2020a`' in out                          # code spans are left alone
    # a paragraph break closes any open parenthesis
    assert bt.cite('(open\n\nlee2019b says', keys).endswith('\\citet{lee2019b} says')
    assert bt.cite('smith2020ab', keys) == 'smith2020ab'   # a longer word is not a key


def test_numbering_matches_latex_and_names_each_discrepancy():
    main = '## 1 Intro\n\n### 1.1 A\n\n### 1.2 B\n\n## 2 Next\n\n## Use of an AI assistant\n'
    appendix = '## Appendix A: Material\n\n### A.1 One\n\n## Appendix B: Tables\n'
    bt.check_numbering(main, appendix)
    with pytest.raises(bt.BuildError) as err:
        bt.check_numbering(main.replace('### 1.2 B', '### 1.3 B'), appendix)
    assert "'1.3 B': written 1.3, LaTeX gives 1.2" in str(err.value)
    stripped = bt.strip_numbers(main + appendix)
    assert '## Intro' in stripped and '### A' in stripped and '## Material' in stripped
    assert '## Use of an AI assistant {-}' in stripped


def test_the_committed_draft_preprocesses():
    title, abstract, body, keys = bt.preprocess(bt.SOURCE.read_text())
    assert title == 'Materials NER on unseen documents: a pre-registered evaluation on three corpora'
    assert abstract.startswith('MaterioMiner, a materials-science')
    assert len(keys) == len(set(keys)) == 55
    assert bt.DROPPED not in body and '## References' not in body and '## Abstract' not in body
    assert body.count('\\appendix') == 1 and '\\nocite{' + bt.COMPANION_KEY + '}' in body
    assert body.index('\\bibliography{') < body.index('## Replication material')
    assert all(f'{{{k}}}' in body or f'{{{k}}}' in abstract for k in keys)   # each key is cited
    assert not re.search(r'(?<![{`\w])(' + '|'.join(map(re.escape, keys)) + r')(?![}`\w])',
                         body)                                             # none left bare
    assert '$S^2/(1-\\bar{\\rho})' in body and 'ρ̄' not in body
    assert re.search(r'^## Use of an AI assistant \{-\}$', body, re.M)
    # no written numbers ("### A uniform decoding bias" is a heading's first word)
    assert not re.search(r'^#{2,3} (\d+(\.\d+)*|[A-Z]\.\d+|Appendix [A-Z]:) ', body, re.M)


def test_style_files_are_the_author_kits():
    bt.check_style()
    for name in bt.STYLE_SHA256:
        assert (bt.STYLE / name).exists()


@pytest.fixture(scope='module')
def tex(tmp_path_factory):
    if not shutil.which('pandoc'):
        pytest.skip('pandoc is not here')
    out = tmp_path_factory.mktemp('tacl')
    result = bt.build(out=out, tex_only=True)
    assert result['pdf'] is None
    return (out / f'{bt.NAME}.tex').read_text()


def test_latex_has_the_style_floats_and_no_longtable(tex):
    assert '\\documentclass[11pt,a4paper]{article}' in tex and '\\usepackage{tacl2021v1}' in tex
    assert 'pdfauthor={}' in tex and '\\author{}' in tex
    assert 'longtable' not in tex
    md = bt.SOURCE.read_text()
    assert tex.count('\\begin{tabular}') == len(re.findall(r'^\*\*Table [A-Z]?\d+\.\*\*', md,
                                                             re.M))
    assert tex.count('\\begin{figure}') == md.count('![Figure')
    assert tex.count('\\caption*{') == tex.count('\\begin{tabular}') + tex.count('\\begin{figure}')
    assert '\\textbf{Table B1.}' in tex and '\\begin{Verbatim}' in tex


def test_pdf_pages_and_properties(tmp_path):
    if not all(shutil.which(t) for t in ('pandoc', 'pdflatex', 'bibtex', 'latexmk', 'pdfinfo')):
        pytest.skip('the LaTeX tools are not here')
    result = bt.build(out=tmp_path)
    assert Path(tmp_path / f'{bt.NAME}.pdf').exists()
    warnings = result['warnings']
    assert warnings['undefined_citations'] == [] and warnings['float_too_large'] == []
    props = result['properties']
    assert props['Author'] == '' and props['Creator'] == '' and props['Producer'] == ''
    assert props['Title'].startswith('Materials NER on unseen documents')
    starts = result['starts']
    assert starts['references'] <= starts['appendix_a'] <= starts['appendix_b'] <= result['pages']
    assert 0 < result['main_text'] < result['pages']
