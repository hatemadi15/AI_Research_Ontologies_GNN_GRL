"""The submission PDFs' HTML (paper_pdf.py): tables kept with their captions and notes, wide ones
on landscape pages, and each figure placed above its legend."""
import pytest

import paper_pdf as pp


def test_tables_and_figures():
    pytest.importorskip('markdown_it')
    md = ('# T\n\n**Table 1.** Narrow.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nIts note.\n\n'
          '**Table 2.** Wide.\n\n| a | b | c | d | e | f |\n|---|---|---|---|---|---|\n'
          '| 1 | 2 | 3 | 4 | 5 | 6 |\n\n## Figure legends\n\n**Figure 1. A title.** The legend.\n')
    body = pp.to_html(md)
    wrapped = pp.wrap_tables(body)
    assert wrapped.count('<section class="tbl">') == 1 and wrapped.count('<section class="tbl wide">') == 1
    first = wrapped.split('<section class="tbl">')[1].split('</section>')[0]
    assert '<strong>Table 1.</strong>' in first and 'Its note.' in first      # caption and note kept
    placed = pp.place_figures(body, '/figs')
    assert '<figure><img src="file:///figs/fig8_method.png"' in placed
    assert '<figcaption><p><strong>Figure 1. A title.</strong> The legend.</p>' in placed
