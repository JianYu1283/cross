"""Convert the finalized paper_draft.md to submission formats (LaTeX/PDF/DOCX)."""
import os
import re
import pypandoc

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MD = os.path.join(ROOT, "paper", "paper_draft.md")
TEX = os.path.join(ROOT, "paper", "paper_draft.tex")
PDF = os.path.join(ROOT, "paper", "paper_draft.pdf")
DOCX = os.path.join(ROOT, "paper", "paper_draft.docx")

with open(MD, "r", encoding="utf-8") as f:
    text = f.read()

# strip the "# Title" heading (title comes from -M metadata; avoids a duplicate title)
text = re.sub(r"^# .*\n", "", text, count=1)
# strip the "**Target:** ..." metadata line (not part of the manuscript)
text = re.sub(r"^\*\*Target:\*\*.*\n", "", text, flags=re.M)
# strip the "---" separator after the author block
text = re.sub(r"^-{3,}\s*\n", "", text, count=1)

clean = os.path.join(ROOT, "paper", "_clean.md")
with open(clean, "w", encoding="utf-8") as f:
    f.write(text)

header = os.path.join(ROOT, "paper", "_header.tex")
with open(header, "w", encoding="utf-8") as f:
    f.write(
        "\\usepackage{amssymb}\n"
        "\\usepackage{newunicodechar}\n"
        "\\newunicodechar{✓}{$\\checkmark$}\n"
        "\\newunicodechar{✗}{$\\times$}\n"
        "\\newunicodechar{≈}{$\\approx$}\n"
    )

meta = [
    "--standalone",
    "--from", "markdown",
    "--pdf-engine=xelatex",
    "--include-in-header", header,
    "-V", "documentclass=article",
    "-V", "fontsize=11pt",
    "-V", "geometry:margin=1in",
    "-V", "colorlinks=true",
    "-M", "title=Low-level Call-aware Heterogeneous Graph Learning for Smart Contract Reentrancy Detection",
    "-M", "date=2026-10-10",
]

pypandoc.convert_file(clean, "latex", outputfile=TEX, extra_args=meta + ["-t", "latex"])
pypandoc.convert_file(clean, "pdf", outputfile=PDF, extra_args=meta + ["-t", "pdf"])
pypandoc.convert_file(clean, "docx", outputfile=DOCX, extra_args=meta + ["-t", "docx"])

os.remove(clean)
os.remove(header)
for p in (TEX, PDF, DOCX):
    print(f"wrote {p} ({os.path.getsize(p)} bytes)")
