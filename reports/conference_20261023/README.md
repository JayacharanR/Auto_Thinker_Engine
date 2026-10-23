# Conference paper — 23 October 2026

The editable deliverable is **Auto_Thinker_Engine_Conference_Paper.docx**.
It follows the supplied IEEE A4 template. The 7-page rendered preview is in
`qa/Auto_Thinker_Engine_Conference_Paper.pdf`.

Only six unfilled fields are bold: author names, department/programme, institution,
city/country, email/ORCID, and optional acknowledgments. All technical text,
results, literature table, metrics, figures and references are filled.

`figures/` contains the original PNG and SVG scientific graphs and architecture.
`source_manifest.json` records the template and project-evidence SHA-256 values
and verified primary reference URLs. `qa/validation.json` records structural,
placeholder and rendered-page checks. `qa/contact_sheet.png` previews all pages.

The input template uses Strict OOXML. `template_normalized.docx` converts its
namespaces and physical-unit dimensions to transitional OOXML for python-docx,
while preserving template styles, A4 page dimensions and conference margins.
`build_paper.py` rebuilds content using python-docx, lxml, matplotlib, NumPy and
Pillow in an environment where those packages are available. It does not run
training or modify experiments.

The paper reports an evidence snapshot, including failed safety/planning gates.
M6 final driving remains pending; the additional positive cache is verified but
has no new fitted-model result in this snapshot.
