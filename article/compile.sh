#!/usr/bin/env bash
# Compile article/main.pdf without a local MacTeX install.
# Uses Docker TeX Live when latexmk is not on PATH.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if command -v latexmk >/dev/null 2>&1; then
  exec latexmk -pdf -interaction=nonstopmode main.tex
fi

if command -v tectonic >/dev/null 2>&1; then
  exec tectonic -X compile main.tex
fi

if command -v docker >/dev/null 2>&1; then
  docker run --rm \
    -v "$ROOT:/work" \
    -w /work \
    texlive/texlive:latest \
    latexmk -pdf -interaction=nonstopmode main.tex
  exit 0
fi

cat >&2 <<'EOF'
No LaTeX toolchain found.

Option A — install BasicTeX (native latexmk), then open a new terminal:
  brew install --cask basictex
  eval "$(/usr/libexec/path_helper)"
  sudo tlmgr update --self
  sudo tlmgr install latexmk collection-fontsrecommended

Option B — install Docker Desktop, then rerun:
  ./compile.sh

Option C — install Tectonic (no sudo):
  brew install tectonic
  tectonic main.tex
EOF
exit 1
