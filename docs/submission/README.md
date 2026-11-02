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

**Status: submitted on 2 November 2026** (the files of `main` at
`325019a`). The steps below are kept as the record; still to come are
the arXiv preprint (optional), the editors' decision and, after
acceptance, the items at the end.

The affiliation (American University of Beirut), the sole authorship, the
ORCID iD and the cover letter are filled in. The AI assistant, Claude
Code, is named once, in a short note in the Methods (*Use of an AI
assistant*), as the author asked; that note is what the journal's policy
on AI requires.

**Done before submission (2 November)**
- The work is merged into `main` as `28effc9` (pull request #16), so the
  paper's repository link opens the submission version. The repository's
  README opens with a pointer to the paper's files, for readers who
  follow that link.
- The ORCID iD, https://orcid.org/0009-0008-8515-6974, is on the cover
  letter.

**In the submission system** (checked against the journal's pages on
2 November)
- Start from the journal's *Submit manuscript* link,
  https://submission.springernature.com/new-submission/41524/3, and
  create a Springer Nature account if you have none.
- Link the ORCID iD under *Personal Profile*, *Create/link an Open
  Researcher Contributor ID (ORCID)*, by signing in to ORCID. The journal
  needs it before the final version; it can be linked at any time
  before acceptance.
- Upload `PAPER_A.pdf`, which holds the text and figures, and
  `PAPER_A_SI.pdf` as the Supplementary Information (the journal asks
  for one merged PDF). Paste `COVER_LETTER.md`.
- The form asks whether the work was discussed with one of the
  journal's editors before submission; unless you contacted one, the
  answer is no. No related manuscript is under consideration elsewhere
  (the companion study is in preparation).
- Ask for the 50% discount on the article processing charge. Lebanon is
  on Springer Nature's list of lower-middle-income locations with a 50%
  discount, which must be requested at submission. The charge, set at
  acceptance, is £2,790 / $3,590 / €3,090 for a research article (plus
  any local tax). AUB Libraries can say whether an institutional
  agreement covers it.
- Suggested and excluded referees: none (the author's call, 2 November).
- The Reporting Summary and the Nature Portfolio editorial policy
  checklist (the editors need the checklist before peer review): the
  answers are in `REPORTING_SUMMARY.md` and the paper's statements (data
  and code availability, competing interests, the use of AI).

**A preprint on arXiv** (optional; the journal's preprint policy allows
it at any time during review)
- One PDF, the paper followed by its SI
  (`pdfunite PAPER_A.pdf PAPER_A_SI.pdf PAPER_A_arXiv.pdf`), uploaded as
  a PDF-only submission, which arXiv accepts because it is not made
  with TeX.
- Primary category cs.CL, cross-listed to cond-mat.mtrl-sci; licence
  CC BY 4.0. A first submission may need an endorsement.
- Once it is announced, tell the journal its DOI (10.48550/arXiv.<ID>)
  and licence, as the preprint policy asks. After publication, add the
  journal reference and DOI to the arXiv entry.

**After acceptance**
- An archived copy with a DOI, which the Code availability statement
  promises:
  1. Sign in to zenodo.org with GitHub and switch the repository on.
  2. Make a GitHub release; Zenodo then assigns the DOI.
  3. Add the DOI to the statement.
- The figure files, uploaded separately (300 dpi PNGs in `docs/figures`).
- The article processing charge, less the discount if it was granted.

**Optional**
- Shortening the main text (about 7,300 words before the Methods; npj
  sets no limit).
- A preprint of the companion study. The paper cites it as "H.M., in
  preparation", which needs nothing more unless it is submitted
  elsewhere first.
