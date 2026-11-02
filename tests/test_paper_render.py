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
        'model. *npj Comput. Mater.* **8**, 102 (2022). https://doi.org/10.1038/s41524-022-00784-w')
    assert pr.format_entry(bib['devlin2019bert']).startswith(
        'Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. BERT: Pre-training')
    assert '4171–4186 (Association for Computational Linguistics, 2019).' in pr.format_entry(bib['devlin2019bert'])
    assert pr.format_entry(bib['nakayama2018seqeval']).endswith(
        'framework. GitHub https://github.com/chakki-works/seqeval (2018).')
    assert pr.format_entry(bib['many2020authors']).startswith('One, A. et al. Many.')


def test_numbering_and_sections():
    bib = pr.parse_bib(BIB)
    assert pr.compress([5, 6]) == '5,6' and pr.compress([3, 1, 2, 7]) == '1–3,7'
    text = ('# T\n\nMatSciBERT (gupta2022matscibert) is BERT (devlin2019bert); seqeval\n'
            '(nakayama2018seqeval; gupta2022matscibert) and (seqeval, nakayama2018seqeval, default).\n\n'
            '## References\n\nplaceholder\n\n## Figure legends\n\nF.\n')
    out = pr.render(text, bib, 'References')
    assert 'MatSciBERT<sup>1</sup> is BERT<sup>2</sup>; seqeval<sup>1,3</sup> and (seqeval,<sup>3</sup>, default).' in out
    assert '## References\n\n1. Gupta, T.' in out and '\n3. Nakayama, H.' in out
    assert 'placeholder' not in out and out.endswith('## Figure legends\n\nF.\n')
    si = pr.render('Note (devlin2019bert).\n', bib, 'Supplementary references')
    assert si.startswith('Note<sup>1</sup>.') and si.endswith('## Supplementary references\n\n1. Devlin, J., '
                                                       'Chang, M.-W., Lee, K. & Toutanova, K. BERT: '
                                                       'Pre-training of Deep Bidirectional Transformers. '
                                                       'In *Proc. NAACL* 4171–4186 '
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
    assert '<!--' not in out and out.index('More<sup>2</sup>.') < out.index('## References')
    assert out.endswith('## Tables\n\n**Table 1.** Corpora<sup>3</sup>.\n\n| a |\n|---|\n| 1 |\n\nIts note.\n')
    assert '\n1. Gupta' in out and '\n2. Devlin' in out and '\n3. Nakayama' in out
    assert 'Text<sup>1</sup>.\n\nMore<sup>2</sup>.' in out      # the block's blank lines collapse
    kept = pr.render(text, bib, 'References')
    assert '<!--' not in kept and kept.index('**Table 1.** Corpora<sup>2</sup>.') < kept.index('More<sup>3</sup>.')
    assert '## Tables' not in kept


def test_preprints_questions_and_no_empty_section():
    """An arXiv preprint reads "Preprint at https://arxiv.org/abs/<id> (year)."; a title ending
    in a question mark gets no full stop; a proceedings paper with an arXiv id and no DOI ends
    with the arXiv link; a file that cites nothing gets no reference section."""
    bib = pr.parse_bib(r"""
@misc{agarwal2020entityswitched,
  author = {Agarwal, Oshin and Nenkova, Ani},
  title  = {Entity-Switched Datasets},
  year   = {2020},
  eprint = {2004.04123},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2004.04123}
}
@article{kim2022biomedical,
  author  = {Kim, Hyunjae and Kang, Jaewoo},
  title   = {How Do Your Models Generalize?},
  journal = {IEEE Access},
  volume  = {10},
  pages   = {31513--31523},
  year    = {2022}
}
@inproceedings{loshchilov2019decoupled,
  author    = {Loshchilov, Ilya and Hutter, Frank},
  title     = {Decoupled Weight Decay Regularization},
  booktitle = {International Conference on Learning Representations},
  year      = {2019},
  eprint    = {1711.05101},
  archivePrefix = {arXiv}
}
""")
    assert pr.format_entry(bib['agarwal2020entityswitched']) == (
        'Agarwal, O. & Nenkova, A. Entity-Switched Datasets. Preprint at '
        'https://arxiv.org/abs/2004.04123 (2020).')
    assert pr.format_entry(bib['kim2022biomedical']) == (
        'Kim, H. & Kang, J. How Do Your Models Generalize? *IEEE Access* **10**, 31513–31523 (2022).')
    assert pr.format_entry(bib['loshchilov2019decoupled']).endswith(
        '(2019). https://arxiv.org/abs/1711.05101')
    assert pr.render('No citations here.\n', bib, 'Supplementary references') == 'No citations here.\n'


def test_journal_abbreviations():
    """Journals take their Nature-style abbreviations; proceedings titles start "Proc."."""
    assert pr.venue_name('Nature Communications') == 'Nat. Commun.'
    assert pr.venue_name('Proceedings of the 58th Annual Meeting') == 'Proc. 58th Annual Meeting'
    assert pr.venue_name('Findings of the ACL') == 'Findings of the ACL'
