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
| `COVER_LETTER.md` | The cover letter, ready to paste. |
| `REPORTING_SUMMARY.md` | The answers to the journal's Reporting Summary form, to copy into it. |

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

## What is left, and when (see also `docs/PAPER_A_NOTES.md`)

The affiliation (American University of Beirut), the AI assistant
(Claude, Anthropic), the sole authorship and the cover letter are filled
in.

**Before you submit**
- Merge this work into `main`. The paper's repository link opens `main`,
  which still holds an early draft until the branch is merged by a pull
  request.
- An ORCID iD (free at orcid.org), which the submission system asks of
  the corresponding author.

**In the submission system**
- Upload `PAPER_A.pdf`, which holds the text and figures, and
  `PAPER_A_SI.pdf` as the Supplementary Information. Paste
  `COVER_LETTER.md`.
- The Reporting Summary: copy the answers from `REPORTING_SUMMARY.md`.
- The editorial policy checklist: it confirms statements the paper
  already makes (data and code availability, competing interests, the
  use of AI).
- Suggested and excluded referees are optional. If you suggest some,
  candidates are authors of the corpora and baseline the paper uses
  (MaterioMiner, MuLMS, SOFC-Exp, MatSciBERT) or of NER
  data-augmentation studies (e.g. Dai and Adel). Check their current
  affiliations and that none has a conflict of interest with you.

**After acceptance**
- An archived copy with a DOI, which the Code availability statement
  promises:
  1. Sign in to zenodo.org with GitHub and switch the repository on.
  2. Make a GitHub release; Zenodo then assigns the DOI.
  3. Add the DOI to the statement.
- The figure files, uploaded separately (300 dpi PNGs in `docs/figures`).

**Optional**
- Shortening the main text (about 7,300 words before the Methods; npj
  sets no limit).
- A preprint of the companion study. The paper cites it as "H.M., in
  preparation", which needs nothing more unless it is submitted
  elsewhere first.
