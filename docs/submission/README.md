# Paper A: the submission files

These files are for *npj Computational Materials*. They are generated from
the sources in `docs/`, so edit the sources and rebuild rather than editing
these files.

| File | What it is |
|---|---|
| `PAPER_A.pdf` | The manuscript for the first submission, one file with the text and figures. Each figure is on its own page with its legend after the references, and the tables come at the end, as the journal asks. |
| `PAPER_A_SI.pdf` | The Supplementary Information as one merged PDF. Wide tables are on landscape pages. |
| `PAPER_A.md`, `PAPER_A_SI.md` | The rendered text: numbered references in Nature style, with the tables moved to the end of the main text. |
| `PAPER_A.html`, `PAPER_A_SI.html` | The pages the PDFs are printed from. |
| `COVER_LETTER.md` | A draft cover letter. The bracketed items are the author's to complete. |

## Rebuild after any edit of `docs/PAPER_A.md`, `docs/PAPER_A_SI.md` or `docs/references.bib`

```
python src/graph/paper_render.py    # numbered references, tables to the end
python src/graph/paper_pdf.py       # HTML and PDFs (needs markdown-it-py and Node Playwright)
```

The figures come from `python src/graph/paper_figures.py` (`docs/figures/`):

| Paper figure | File |
|---|---|
| Figure 1 | `fig8_method.png` |
| Figure 2 | `fig9_units.png` |
| Figure 3 | `fig7_names.png` |
| Figure 4 | `fig10_recall_matched.png` |

## Still the author's (see also `docs/PAPER_A_NOTES.md`)

**For the manuscript**
- The affiliation on the title page.
- The name and version of the AI assistant in the AI-use statement.
- If true, that the author critically revised and approved the final text (Author contributions).
- The commit this paper describes tagged, and an archived copy with a DOI deposited and cited in the Code availability statement.
- The companion study: either posted as a preprint, or uploaded as a related manuscript.

**For the submission system**
- The cover letter completed: affiliation and contact, related manuscript, and suggested and excluded referees.
- The Reporting Summary and the editorial policy checklist, which are encouraged at submission and required after review.
- The figure files uploaded separately. They are 300 dpi PNGs, and the journal asks for separate files at the final stage.
