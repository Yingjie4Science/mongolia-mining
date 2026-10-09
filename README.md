# mongolia-mining

## Mongolia mining, land-cover, and ecosystem-service research

This repository supports reproducible work on mining footprints in Mongolia,
land-use/land-cover (LULC) change, and ecosystem-service footprint summaries.
It is deliberately accompanied by a small, platform-independent project-memory
system so that work can continue across computers, AI tools, and accounts
without treating a chat history as the source of truth.

### Status

The repository contains two mining-polygon datasets and checked-in scripts for
area summaries, LULC change, and ecosystem-service footprinting.  Some
area-summary CSVs and selected figures are present.  The LULC and
ecosystem-service analytical workflows have **not** been re-run or
scientifically validated as part of this documentation update.  Read the
status labels in [`memory/PROJECT_CONTEXT.md`](memory/PROJECT_CONTEXT.md)
before using any result.

### Start here

1. Read [`memory/PROJECT_CONTEXT.md`](memory/PROJECT_CONTEXT.md) for scope,
   verified evidence, and known limits.
2. Read [`memory/RESEARCH_DECISIONS.md`](memory/RESEARCH_DECISIONS.md) before
   changing spatial methods or interpreting outputs.
3. Select a current task from [`memory/NEXT_STEPS.md`](memory/NEXT_STEPS.md).
4. Consult the focused workflow notes in [`code/LULC_ANALYSIS_README.md`](code/LULC_ANALYSIS_README.md)
   and [`code/QUICKSTART.md`](code/QUICKSTART.md), then verify inputs and
   outputs in the working environment.

### Repository guide

| Location | Purpose | Evidence status |
| --- | --- | --- |
| `code/` | Python, R Markdown, and notebook workflows | Source code; execution must be verified per run |
| `data/` | Local input data and locally generated outputs | Intentionally ignored by Git; may be restricted |
| `figures/` | Maps and other visual outputs | Files may be present; inspect provenance before reuse |
| `memory/` | Portable context, decisions, log, next steps, and data dictionary | Maintained project knowledge base |
| `AGENTS.md` / `CLAUDE.md` | Cross-assistant operating instructions | Read before substantial work |

### Reproducing an analysis safely

Use the relevant script as a starting point, but do not assume its configured
Windows path, dataset selection, or documented example output applies to the
current machine.  Before running a spatial comparison:

1. Confirm that the required vector, raster, and boundary files exist.
2. Inspect CRS, pixel grid, nodata, class codes, and geometry validity.
3. Run a small, logged test first.
4. Preserve raw inputs and write new outputs to a dated run directory.
5. Inspect tables and rendered figures before reporting results.

### Data and publication safety

The `data/` directory is ignored by Git.  Do not commit credentials, API keys,
private mine-level data, restricted source documents, or large raw rasters.
Use relative paths in shareable code and documentation.  Keep public claims
limited to results with traceable inputs, configuration, and validation.

### Portable use

To continue elsewhere, copy or clone the repository and provide an assistant
with `README.md`, `AGENTS.md`, `CLAUDE.md`, and the `memory/` directory.  Copy
only data that may lawfully be shared.  The Markdown memory is authoritative
for documented context; platform memory and chat histories are convenience
layers, not archival records.
