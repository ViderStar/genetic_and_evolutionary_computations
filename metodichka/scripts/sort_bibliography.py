"""Упорядочивает sections/references.tex по первому упоминанию источника в тексте (СТП БГУИР).

Порядок берётся из main.aux, поэтому сначала соберите документ:
    latexmk -pdf main.tex && python3 scripts/sort_bibliography.py && latexmk -pdf main.tex
Непроцитированные источники переносятся в конец списка и выводятся в консоль.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
aux = (ROOT / "main.aux").read_text(encoding="utf-8", errors="replace")
order = []
for group in re.findall(r"\\citation\{([^}]*)\}", aux):
    for key in (k.strip() for k in group.split(",")):
        if key not in order:
            order.append(key)

path = ROOT / "sections" / "references.tex"
text = path.read_text(encoding="utf-8")
start, end = text.index("\\bibitem"), text.index("\\end{thebibliography}")
items = {re.match(r"\\bibitem\{([^}]*)\}", chunk).group(1): chunk.rstrip() + "\n\n"
         for chunk in re.split(r"(?=\\bibitem\{)", text[start:end]) if chunk.strip()}
uncited = [k for k in items if k not in order]
body = "".join(items[k] for k in order if k in items) + "".join(items[k] for k in uncited)
path.write_text(text[:start] + body + text[end:], encoding="utf-8")
print("порядок:", ", ".join(k for k in order if k in items))
if uncited:
    print("не процитированы:", ", ".join(uncited))
