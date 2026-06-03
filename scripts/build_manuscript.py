"""Build the manuscript Word document (and a companion PDF) reproducibly.

Pipeline:  results/manuscript/manuscript.md  --pandoc-->  FACE_trans_diagnostic_v2.docx
  + two OOXML post-patches that make the file fully schema-valid:
    (1) declare the png content type in [Content_Types].xml (pandoc omits it for embedded figures);
    (2) drop the redundant <m:sty m:val="p"/> from the 9 <m:nor/>+<m:sty/> math runs (a pandoc OMML
        quirk that some strict validators reject; <m:nor/> alone already renders operators upright).

Figures must exist first:  python3 scripts/figures_manuscript.py
Requires: pandoc (math -> editable OMML). Optional companion PDF: xelatex + STIX Two Math.

Usage:  python3 scripts/build_manuscript.py [--pdf]
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAN = ROOT / "results" / "manuscript"
MD = MAN / "manuscript.md"
DOCX = MAN / "FACE_trans_diagnostic_v2.docx"
PDF = MAN / "FACE_trans_diagnostic_v2.pdf"

PANDOC = ["pandoc", str(MD), "-o", str(DOCX), f"--resource-path={ROOT}",
          "--toc", "--toc-depth=2", "--number-sections", "--metadata", "lang=en"]

PNG_DECL = '<Default Extension="png" ContentType="image/png"/>'
NORSTY = '<m:rPr><m:nor /><m:sty m:val="p" /></m:rPr>'
NOR = '<m:rPr><m:nor /></m:rPr>'


def patch_docx(path: Path) -> None:
    zin = zipfile.ZipFile(path)
    items = {n: zin.read(n) for n in zin.namelist()}
    zin.close()
    ct = items["[Content_Types].xml"].decode("utf8")
    if 'Extension="png"' not in ct:
        items["[Content_Types].xml"] = ct.replace("</Types>", PNG_DECL + "</Types>").encode("utf8")
    doc = items["word/document.xml"].decode("utf8")
    n = doc.count(NORSTY)
    items["word/document.xml"] = doc.replace(NORSTY, NOR).encode("utf8")
    tmp = path.with_suffix(".tmp.docx")
    zo = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
    for k, d in items.items():
        zo.writestr(k, d)
    zo.close()
    tmp.replace(path)
    print(f"  patched: png content-type + {n} OMML nor/sty runs")


def main() -> None:
    if not MD.exists():
        sys.exit(f"missing {MD}")
    if not list((ROOT / "results" / "reports" / "figures").glob("fig*.png")):
        sys.exit("figures missing — run scripts/figures_manuscript.py first")
    print("pandoc -> docx ...")
    subprocess.run(PANDOC, check=True)
    patch_docx(DOCX)
    print(f"  -> {DOCX}  ({DOCX.stat().st_size // 1024} KB)")

    if "--pdf" in sys.argv:
        print("xelatex -> companion pdf ...")
        try:
            subprocess.run(["pandoc", str(MD), "-o", str(PDF), f"--resource-path={ROOT}", "--toc", "--toc-depth=2",
                            "--number-sections", "--pdf-engine=xelatex",
                            "-V", "geometry:margin=0.85in", "-V", "fontsize=10pt",
                            "-V", "mainfont=Times New Roman", "-V", "mathfont=STIX Two Math"],
                           check=True)
            print(f"  -> {PDF}")
        except Exception as e:  # noqa
            print(f"  (pdf skipped: {e})")
    print("done.")


if __name__ == "__main__":
    main()
