"""Render an Arabic markdown guide to a self-contained right-to-left HTML page.

Markdown carries no direction information, so a rendered page is the only way
to get correct RTL in every viewer. Latin identifiers and code stay LTR.

    python3 tools/md2rtlhtml.py in.md out.html "Page title"
"""
import re
import sys
from pathlib import Path

import markdown

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
TITLE = sys.argv[3] if len(sys.argv) > 3 else SRC.stem

text = SRC.read_text()
# drop an RTL wrapper div if the markdown already carries one
text = re.sub(r'^<div dir="rtl"[^>]*>\s*', '', text)
text = re.sub(r'\s*</div>\s*$', '', text)

body = markdown.markdown(text, extensions=['tables', 'sane_lists'])

CSS = """
:root { --ink:#1A1F2E; --blue:#065A82; --teal:#1C7293; --muted:#5A6470;
        --rule:#D6DEE4; --card:#F7F9FB; --amber:#D4922A; --red:#C8463D; }
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { direction: rtl; text-align: right;
       font-family: "Segoe UI", Tahoma, "Noto Naskh Arabic", "Amiri", Arial, sans-serif;
       font-size: 17px; line-height: 1.95; color: var(--ink);
       max-width: 46rem; margin: 0 auto; padding: 2.2rem 1.4rem 5rem; background:#fff; }
h1 { font-size: 1.85rem; color: var(--blue); line-height: 1.5;
     margin: 2.6rem 0 1rem; padding-bottom: .5rem; border-bottom: 3px solid var(--blue); }
h1:first-child { margin-top: 0; }
h2 { font-size: 1.3rem; color: var(--blue); margin: 2.2rem 0 .7rem; line-height: 1.6; }
h3 { font-size: 1.08rem; color: var(--teal); margin: 1.6rem 0 .5rem; }
p { margin: .8rem 0; }
strong { color: #000; font-weight: 700; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 2.4rem 0; }
ul, ol { padding-right: 1.6rem; padding-left: 0; margin: .8rem 0; }
li { margin: .45rem 0; }
blockquote { margin: 1.1rem 0; padding: .9rem 1.1rem .9rem 1rem; background: var(--card);
             border-right: 4px solid var(--teal); border-radius: 3px; }
blockquote p { margin: .45rem 0; }
table { border-collapse: collapse; width: 100%; margin: 1.2rem 0; font-size: .95rem; }
th, td { border: 1px solid var(--rule); padding: .55rem .7rem; text-align: right;
         vertical-align: top; }
th { background: var(--blue); color: #fff; font-weight: 700; }
tr:nth-child(even) td { background: var(--card); }
code { direction: ltr; unicode-bidi: isolate; display: inline-block;
       font-family: "Consolas", "SFMono-Regular", "Courier New", monospace;
       font-size: .88em; background: #EEF2F5; padding: .1rem .34rem; border-radius: 3px; }
pre { direction: ltr; text-align: left; unicode-bidi: isolate; background: var(--card);
      border: 1px solid var(--rule); border-radius: 4px; padding: .9rem; overflow-x: auto; }
pre code { background: none; padding: 0; display: block; }
@media print {
  body { font-size: 11.5pt; max-width: none; padding: 0; }
  h1 { page-break-after: avoid; } h2, h3 { page-break-after: avoid; }
  blockquote, table { page-break-inside: avoid; }
}
"""

OUT.write_text(
    '<!DOCTYPE html>\n<html lang="ar" dir="rtl">\n<head>\n'
    '<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    f'<title>{TITLE}</title>\n<style>{CSS}</style>\n</head>\n<body>\n'
    f'{body}\n</body>\n</html>\n'
)
print(f'{OUT}  ({len(body)} chars of HTML)')
