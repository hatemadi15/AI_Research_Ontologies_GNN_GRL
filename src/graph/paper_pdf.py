"""
paper_pdf.py - the submission PDFs of paper A from its rendered Markdown

npj Computational Materials asks for one PDF (or Word file) with the text and figures at first
submission, each figure legend on the same page as its figure, the tables at the end, and the
Supplementary Information as one merged PDF. This script reads the rendered copies that
paper_render.py writes (docs/submission/PAPER_A.md and PAPER_A_SI.md), converts them to HTML
(markdown-it-py, with tables), puts each figure above its legend on a page of its own, sets wide
tables (six columns or more) on landscape pages, and prints both files to PDF with Chromium
(Playwright, through Node). The HTML files are kept beside the PDFs.

Usage:
  python src/graph/paper_render.py && python src/graph/paper_pdf.py [--out docs/submission]
"""

import argparse
import html
import json
import os
import re
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIGURES = {1: 'fig8_method.png', 2: 'fig9_units.png', 3: 'fig7_names.png',
           4: 'fig10_recall_matched.png'}
WIDE = 6          # tables with at least this many columns go on landscape pages

CSS = """
@page { size: A4; margin: 20mm 20mm 18mm 20mm; }
@page wide { size: A4 landscape; margin: 14mm 14mm 14mm 14mm; }
html { font-family: 'Liberation Serif', 'Times New Roman', serif; font-size: 10.5pt; }
body { line-height: 1.42; color: #111; margin: 0; }
h1 { font-size: 16pt; line-height: 1.25; margin: 0 0 8pt; }
h2 { font-size: 12.5pt; margin: 16pt 0 6pt; break-after: avoid; }
h3 { font-size: 11pt; font-style: italic; margin: 12pt 0 4pt; break-after: avoid; }
p, li { text-align: justify; hyphens: auto; }
p { margin: 0 0 6pt; }
ul, ol { margin: 0 0 6pt 0; padding-left: 16pt; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.5pt; }
table { border-collapse: collapse; width: 100%; font-family: 'Liberation Sans', 'DejaVu Sans',
        sans-serif; font-size: 7.4pt; line-height: 1.22; margin: 4pt 0 8pt; }
th, td { border-top: 0.4pt solid #aaa; padding: 1.6pt 3pt; vertical-align: top; text-align: left; }
thead th { border-top: 1pt solid #000; border-bottom: 0.8pt solid #000; }
tbody tr:last-child td { border-bottom: 1pt solid #000; }
section.tbl { break-inside: auto; margin-bottom: 10pt; }
section.tbl.wide { page: wide; }
section.tbl > p { font-size: 9pt; line-height: 1.3; }
figure { break-before: page; break-inside: avoid; margin: 0; }
figure img { display: block; max-width: 100%; max-height: 175mm; margin: 0 auto 6mm; }
figure figcaption p { font-size: 9.5pt; }
h2.tables { break-before: page; }
h2.legends { break-before: page; }
figure.first { break-before: avoid; }
section.tbl > p:first-child { break-inside: avoid; break-after: avoid; }
sup { font-size: 70%; line-height: 0; vertical-align: super; }
"""


def to_html(markdown_text):
    """HTML from Markdown (CommonMark with tables; inline HTML kept, for the superscript
    citations that paper_render.py writes)."""
    from markdown_it import MarkdownIt
    return MarkdownIt('commonmark', {'html': True}).enable('table').render(markdown_text)


def columns(table_html):
    head = re.search(r'<tr>(.*?)</tr>', table_html, re.S)
    return len(re.findall(r'<th', head.group(1))) if head else 0


def wrap_tables(body):
    """Each table with its caption (the nearest preceding "Table N." paragraph) and any
    paragraphs up to the next caption or heading, in a <section class="tbl">; wide ones get
    landscape pages. A table with no caption of its own (the second part of a table) gets
    "Table N (continued)."."""
    out, pos, last = [], 0, None
    for m in re.finditer(r'<table>.*?</table>\n', body, re.S):
        cap = body.rfind('<p><strong>Table ', pos, m.start())
        start = cap if cap >= 0 else m.start()
        number = re.match(r'<p><strong>Table (S?\d+)\.', body[cap:]) if cap >= 0 else None
        last = number.group(1) if number else last
        cont = (f'<p><strong>Table {last} (continued).</strong></p>\n'
                if cap < 0 and last else '')
        end = m.end()
        nxt = re.search(r'<p><strong>Table |<h[1-3]|<table>', body[end:])
        tail = body[end:end + (nxt.start() if nxt else 0)]
        if tail.strip().startswith('<p>') and '<p><strong>Table ' not in tail:
            end += len(tail)                     # the table's note, up to the next caption
        cls = 'tbl wide' if columns(m.group(0)) >= WIDE else 'tbl'
        out.append(body[pos:start])
        out.append(f'<section class="{cls}">\n{cont}{body[start:end]}</section>\n')
        pos = end
    out.append(body[pos:])
    return ''.join(out)


def place_figures(body, figures_dir):
    """The figure legends, each below its figure on a page of its own."""
    def figure(m):
        n = int(m.group(1))
        src = os.path.join(figures_dir, FIGURES[n]) if n in FIGURES else None
        img = f'<img src="file://{html.escape(src)}" alt="Figure {n}">' if src else ''
        return f'<figure>{img}<figcaption>{m.group(0)}</figcaption></figure>\n'
    body = re.sub(r'<p><strong>Figure (\d+)\. .*?</p>\n', figure, body, flags=re.S)
    return body.replace('<figure>', '<figure class="first">', 1)


def page(title, body):
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<title>{html.escape(title)}</title><style>{CSS}</style></head>'
            f'<body>\n{body}</body></html>\n')


def build(md_path, html_path, figures_dir, main):
    with open(md_path) as f:
        text = f.read()
    body = to_html(text)
    if main:
        body = body.replace('<h2>Tables</h2>', '<h2 class="tables">Tables</h2>')
        body = body.replace('<h2>Figure legends</h2>', '<h2 class="legends">Figure legends</h2>')
        body = place_figures(body, figures_dir)
    body = wrap_tables(body)
    title = re.search(r'<h1>(.*?)</h1>', body)
    with open(html_path, 'w') as f:
        f.write(page(re.sub('<[^>]+>', '', title.group(1)) if title else 'Paper A', body))


NODE = """
const { chromium } = require('playwright');
(async () => {
  const jobs = JSON.parse(process.argv[1]);
  const browser = await chromium.launch();
  const page = await browser.newPage();
  for (const [src, out] of jobs) {
    await page.goto('file://' + src, { waitUntil: 'load' });
    await page.pdf({ path: out, printBackground: true, preferCSSPageSize: true,
      displayHeaderFooter: true, headerTemplate: '<span></span>',
      footerTemplate: '<div style="font-size:7pt;width:100%;text-align:center;color:#666">' +
        '<span class="pageNumber"></span> / <span class="totalPages"></span></div>' });
  }
  await browser.close();
})();
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paper A's submission PDFs")
    parser.add_argument('--out', default=os.path.join(ROOT, 'docs', 'submission'))
    parser.add_argument('--figures', default=os.path.join(ROOT, 'docs', 'figures'))
    parser.add_argument('--html-only', action='store_true')
    args = parser.parse_args(argv)
    out, figures = os.path.abspath(args.out), os.path.abspath(args.figures)
    jobs = []
    for name, main_text in (('PAPER_A', True), ('PAPER_A_SI', False)):
        md, page_html = os.path.join(out, name + '.md'), os.path.join(out, name + '.html')
        build(md, page_html, figures, main_text)
        jobs.append([page_html, os.path.join(out, name + '.pdf')])
        print('wrote', page_html)
    if not args.html_only:
        subprocess.run(['node', '-e', NODE, json.dumps(jobs)], check=True)
        for _, pdf in jobs:
            print('wrote', pdf)


if __name__ == '__main__':
    main()
