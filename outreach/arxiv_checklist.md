# arXiv submission checklist

Checked against the current `writeup/paper.tex` + `writeup/refs.bib`. Nothing
here has been submitted yet — this is prep, not a record of a completed
submission.

## 1. The endorsement hurdle (deal with this first, it can block everything else)

arXiv requires new authors without a submission history in a category to be
**endorsed** by an existing arXiv author in that category. As an independent
submitter with zero prior arXiv papers, this will very likely apply.

- Primary category will be `cs.CV` (person detection, architecture ablation) —
  check arxiv.org's current endorsement page for `cs.CV` for the exact
  process; it's usually: arXiv emails a candidate endorser you name (someone
  who has published in `cs.CV`), and they approve via a link.
- **This is the one place the professor-outreach work and this checklist
  connect:** if any professor you email responds positively and has their own
  arXiv papers in a relevant category, asking them to endorse the submission
  is a reasonable, normal ask — but only after genuine interest, not as the
  opening request.
- Don't wait on this to start outreach — the one-pager and `writeup/paper.pdf`
  work fine as plain attachments/links in the meantime. arXiv is the
  upgrade from "attached PDF" to "citable preprint," not a prerequisite.

## 2. Category selection

- **Primary: `cs.CV`** (Computer Vision and Pattern Recognition) — the paper
  is fundamentally a detection pipeline + empirical study.
- **Cross-list candidates:** `cs.LG` (quantization/compression angle) or
  `eess.IV`. Pick one cross-list, not both — over-categorizing reads as
  unfocused to moderators.
- Avoid `cs.AR` (hardware architecture) as primary — the paper measures
  feasibility on existing hardware, it doesn't design hardware; using it as
  primary would misrepresent the contribution to that category's readers.

## 3. Format the submission bundle (source, not just the PDF)

arXiv recompiles from **LaTeX source**, not from an uploaded PDF — bundle:
- `paper.tex`
- `refs.bib`
- `figures/flash_by_variant.pdf`, `figures/stage2_training_curves.pdf`

**Generate and include a `.bbl` file rather than relying on arXiv to run
BibTeX itself** — version differences between local and arXiv's TeX Live can
silently change citation formatting. Locally:
```bash
cd writeup
tectonic --keep-intermediates paper.tex   # leaves paper.bbl alongside paper.pdf
```
Include that `paper.bbl` in the upload bundle instead of (or alongside)
`refs.bib`. Don't commit `.bbl`/`.aux`/`.out` to the repo itself — these are
now in `.gitignore`; generate them fresh right before zipping the arXiv
upload.

**Verify compilation under plain pdfLaTeX, not just tectonic.** arXiv's
default engine is pdfLaTeX unless you explicitly request XeLaTeX/LuaLaTeX in
the submission settings. The current preamble (`geometry`, `graphicx`,
`booktabs`, `amsmath`, `enumitem`, `caption`, `hyperref`) has no
XeLaTeX-specific dependencies, so this should be a non-issue, but confirm
once with a local `pdflatex` + `bibtex` pass before uploading — cheaper to
catch a package incompatibility locally than in arXiv's submission queue.

## 4. Content checks specific to this paper

- **Abstract length: 1,740 characters currently** (223 words). arXiv's cap is
  2,000 — under the limit, but close enough that any future edit to the
  abstract should recheck this.
- **Author block footnote links the GitHub repo** — fine as-is; arXiv doesn't
  require institutional affiliation for independent submissions.
- **Figures are vector PDFs already** (`make_figures.py` output) — no raster/
  DPI concerns.
- Re-read the Limitations section (§5) once more before submission specifically
  for the self-referential-mAP disclosure — it needs to survive verbatim; this
  is the sentence most likely to get quoted back at you in a reviewer-style
  reply from whoever reads it next, and it should already correctly represent
  the honest limitation.

## 5. License

Pick a license at submission time. **Recommend CC BY 4.0** — maximizes
shareability (what you want for outreach), requires only attribution, and is
one of arXiv's standard offered options.

## 6. Submit, then propagate the link

After arXiv processes the submission (typically announced the next business
day), update, in this order:
1. `outreach/one_pager.md` / `.tex` — replace the GitHub-only link with the
   arXiv abstract page URL as the primary citation.
2. `outreach/email_templates.md` — same replacement in `{{repo_or_arxiv_link}}`.
3. `Readme.md` — add the arXiv link next to the existing `writeup/paper.pdf`
   reference.
4. Any email threads already in flight with interested professors — a short
   "now up on arXiv: <link>" follow-up is a legitimate, low-friction reason
   to re-surface a thread that went quiet.
