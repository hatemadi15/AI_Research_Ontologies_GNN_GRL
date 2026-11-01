"""Rendering paper A for submission (paper_render.py): the bibliography parser, the reference
style, the numbering by first citation and the reference sections."""
import paper_render as pr

BIB = r"""
@article{gupta2022matscibert,
  author  = {Gupta, Tanishq and Zaki, Mohd and Krishnan, N. M. Anoop and {Mausam}},
  title   = {{MatSciBERT}: A materials domain language model},
  journal = {npj Computational Materials},
  volume  = {8},
  pages   = {102},
  year    = {2022},
  doi     = {10.1038/s41524-022-00784-w}
}
@inproceedings{devlin2019bert,
  author    = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  title     = {{BERT}: Pre-training of Deep Bidirectional Transformers},
  booktitle = {Proceedings of NAACL},
  pages     = {4171--4186},
  year      = {2019},
  publisher = {Association for Computational Linguistics}
}
@misc{nakayama2018seqeval,
  author = {Nakayama, Hiroki},
  title  = {{seqeval}: A {Python} framework},
  year   = {2018},
  url    = {https://github.com/chakki-works/seqeval}
}
@article{many2020authors,
  author  = {One, A. and Two, B. and Three, C. and Four, D. and Five, E. and Six, F.},
  title   = {Many},
  journal = {J},
  year    = {2020}
}
"""


def test_parse_and_style():
    bib = pr.parse_bib(BIB)
    assert set(bib) == {'gupta2022matscibert', 'devlin2019bert', 'nakayama2018seqeval', 'many2020authors'}
    assert pr.format_entry(bib['gupta2022matscibert']) == (
        'Gupta, T., Zaki, M., Krishnan, N. M. A. & Mausam. MatSciBERT: A materials domain language '
        'model. *npj Computational Materials* **8**, 102 (2022). https://doi.org/10.1038/s41524-022-00784-w')
    assert pr.format_entry(bib['devlin2019bert']).startswith(
        'Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. BERT: Pre-training')
    assert '4171–4186 (Association for Computational Linguistics, 2019).' in pr.format_entry(bib['devlin2019bert'])
    assert pr.format_entry(bib['nakayama2018seqeval']).endswith('(2018). https://github.com/chakki-works/seqeval')
    assert pr.format_entry(bib['many2020authors']).startswith('One, A. et al. Many.')


def test_numbering_and_sections():
    bib = pr.parse_bib(BIB)
    assert pr.compress([5, 6]) == '5,6' and pr.compress([3, 1, 2, 7]) == '1–3,7'
    text = ('# T\n\nMatSciBERT (gupta2022matscibert) is BERT (devlin2019bert); seqeval\n'
            '(nakayama2018seqeval; gupta2022matscibert) and (seqeval, nakayama2018seqeval, default).\n\n'
            '## References\n\nplaceholder\n\n## Figure legends\n\nF.\n')
    out = pr.render(text, bib, 'References')
    assert 'MatSciBERT [1] is BERT [2]; seqeval\n[1,3] and (seqeval, [3], default).' in out
    assert '## References\n\n1. Gupta, T.' in out and '\n3. Nakayama, H.' in out
    assert 'placeholder' not in out and out.endswith('## Figure legends\n\nF.\n')
    si = pr.render('Note (devlin2019bert).\n', bib, 'Supplementary references')
    assert si.startswith('Note [1].') and si.endswith('## Supplementary references\n\n1. Devlin, J., '
                                                       'Chang, M.-W., Lee, K. & Toutanova, K. BERT: '
                                                       'Pre-training of Deep Bidirectional Transformers. '
                                                       'In *Proceedings of NAACL* 4171–4186 '
                                                       '(Association for Computational Linguistics, 2019).\n')


def test_latex_accents_become_unicode():
    assert pr.plain('Gr{\\"u}newald') == 'Grünewald'
    assert pr.plain("S{\\o}gaard and Jos{\\'e}") == 'Søgaard and José'
    assert pr.short_name('Gr{\\"u}newald, Stefan') == 'Grünewald, S.'
    assert pr.plain('Speech \\& Language') == 'Speech & Language'



def test_tables_move_to_the_end():
    bib = pr.parse_bib(BIB)
    text = ('# T\n\nText (gupta2022matscibert).\n\n<!-- table -->\n\n**Table 1.** Corpora '
            '(nakayama2018seqeval).\n\n| a |\n|---|\n| 1 |\n\nIts note.\n\n<!-- /table -->\n\n'
            'More (devlin2019bert).\n\n## References\n\nx\n\n## Figure legends\n\nF.\n')
    out = pr.render(text, bib, 'References', move_tables=True)
    assert '<!--' not in out and out.index('More [2].') < out.index('## References')
    assert out.endswith('## Tables\n\n**Table 1.** Corpora [3].\n\n| a |\n|---|\n| 1 |\n\nIts note.\n')
    assert '\n1. Gupta' in out and '\n2. Devlin' in out and '\n3. Nakayama' in out
    assert 'Text [1].\n\nMore [2].' in out                      # the block's blank lines collapse
    kept = pr.render(text, bib, 'References')
    assert '<!--' not in kept and kept.index('**Table 1.** Corpora [2].') < kept.index('More [3].')
    assert '## Tables' not in kept
