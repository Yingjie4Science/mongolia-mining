# Working instructions for AI assistants

## Purpose

This repository studies mining footprints, land-use/land-cover (LULC) change,
and ecosystem-service footprint metrics in Mongolia.  Treat the repository and
its versioned documentation as the project record; do not infer scientific
findings from chat history, draft scripts, filenames, or example output.

## Required reading order

Before substantial analysis or editing, read:

1. `README.md`
2. `memory/PROJECT_CONTEXT.md`
3. `memory/RESEARCH_DECISIONS.md`
4. `memory/NEXT_STEPS.md`
5. The workflow-specific document in `code/`.

Read `memory/DATA_DICTIONARY.md` before working with data and
`memory/WORK_LOG.md` before resuming an unfinished workflow.

## Evidence and research rules

- Label statements as **verified**, **implemented but unverified**,
  **preliminary**, or **proposed**. Do not convert one status into another
  without evidence.
- Verify the current files, code, configuration, and rendered output before
  saying an analysis succeeded or a figure is ready for use.
- Preserve raw inputs and existing source records. Do not overwrite data,
  generated outputs, or user edits without explicit review and permission.
- Validate CRS, geometry validity, raster alignment (CRS, transform,
  dimensions, resolution, and nodata), and LULC class codes before pixel-wise
  comparisons or area calculations.
- Use an appropriate equal-area CRS for area measurement. Document the CRS and
  whether polygon overlap has been dissolved or double-counted.
- Do not make causal or mining-attribution claims from spatial overlap alone.

## Compute and coding rules

- Avoid loading country-scale rasters or a country-wide mining-mask bounding
  box into memory without a demonstrated resource estimate. Prefer
  polygon-/tile-wise processing and aggregate only the necessary counts.
- Use repository-relative paths in portable code. Never add a personal machine
  path, credential, token, or confidential file name to shareable documents.
- Keep a small test run separate from full outputs. Record commands,
  environment/dependency versions, inputs, and output locations for any run
  used in reporting.
- Treat notebooks and R Markdown files as code that must be executed and
  rendered; their presence alone is not validation.

## Handoff discipline

After substantive, verified work:

1. Update `memory/WORK_LOG.md` with what was run and what was inspected.
2. Add or revise a decision in `memory/RESEARCH_DECISIONS.md` when a method,
   assumption, or interpretation changes.
3. Update `memory/NEXT_STEPS.md` to reflect the next safe action.
4. Update `memory/DATA_DICTIONARY.md` if inputs, schema, or access conditions
   change.

Keep entries concise, dated, and free of sensitive information.
