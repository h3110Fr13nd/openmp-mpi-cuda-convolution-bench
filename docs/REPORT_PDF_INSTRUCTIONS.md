# Report PDF Generation (Recommended)

This project’s Markdown report renders best to PDF using LaTeX (XeLaTeX). The steps below are the most reliable and preserve math and figures cleanly.

## Prerequisites

- `pandoc`
- XeLaTeX (from TeX Live): `texlive-latex-recommended`, `texlive-latex-extra`, `texlive-xetex`

On Ubuntu:

- `sudo apt-get update`
- `sudo apt-get install -y pandoc texlive-latex-recommended texlive-latex-extra texlive-xetex`

## Generate PDF (Recommended Path)

From the repository root:

1. Generate the LaTeX file:

   `pandoc REPORT.md --standalone -o REPORT.tex`

2. Compile to PDF:

   `xelatex -interaction=nonstopmode REPORT.tex`

This produces `REPORT.pdf` in the project root.

## Notes

- This path keeps math rendering consistent and avoids browser headers/footers.
- If you re-run after editing `REPORT.md`, repeat both steps.
