# Paper

JBHI submission, built on the **official IEEEtran journal template**
(`bare_jrnl.tex`, Michael Shell), which is what IEEE distributes for its Transactions
and Journals. `IEEEtran.cls` and `IEEEtran.bst` are vendored so the source compiles
without a TeX Live IEEE package.

```
main.tex        the paper
refs.bib        bibliography (placeholder entries)
bare_jrnl.tex   the official template, kept for reference
IEEEtran.cls    class file, from CTAN
IEEEtran.bst    IEEE bibliography style
figs/           figures
```

Build:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

**This source has never been compiled.** The machine it was written on has no LaTeX
toolchain (`pdflatex`, `xelatex` and `latexmk` are all absent), so spacing, float
placement and overfull boxes are unverified. Compile before reading it as final.

`ieee.org` could not be reached for the author kit --- it answers automated requests
with HTTP 202 and an empty body --- so the class and template come from CTAN, which is
the same package.
