# Project context

**Project:** Mongolia mining — mining footprints, LULC change, and
ecosystem-service footprint analysis

**Purpose of this file:** concise, portable context for a new researcher or AI
assistant.  It is a research handoff, not a results publication.  Last
reviewed: 2026-10-09.

## Research questions

1. What mining polygons are represented in the two available mining datasets?
2. How did LULC composition and transitions change between 2000 and 2020
   within selected mining footprints?
3. What ecosystem-service metrics occur within those footprints, and how do
   results differ by mining dataset and service?

## Evidence-status convention

| Label | Meaning |
| --- | --- |
| **Verified in repository** | A current local file or source configuration was inspected on 2026-10-09. It is not automatically a scientific validation. |
| **Implemented, unverified** | Code/docs exist, but this handoff did not execute and inspect the workflow end-to-end. |
| **Preliminary / unresolved** | A prior working note or draft exists; it must not be reported as a finding. |
| **Proposed** | A recommended next action or interpretation, not completed work. |

## Current evidence snapshot

| Component | What is currently documented | Status |
| --- | --- | --- |
| Mining polygons | `Maus2022` and `Tang2023` GeoPackages are present under `data/mining_polygons_clipped/`. | **Verified in repository** |
| Mining-area summaries | CSV summaries are present: Maus2022 reports 429 features and 1,727.226 km² summed area; Tang2023 reports 557 features and 466.581 km² summed area. | **Verified as existing CSV values; not independently recomputed** |
| Area maps | Point-map PNGs for both datasets are present in `figures/`. | **Files verified; provenance and visual QA not rechecked** |
| LULC workflow | `code/lulc_change_analysis.py` defines a 14-class, 2000–2020 analysis and dated CSV/PNG outputs. | **Implemented, unverified** |
| LULC inputs/results | A LULC archive is present, but extracted `2000.tif`/`2020.tif` availability and any full-run outputs were not confirmed in this handoff. | **Unresolved** |
| Ecosystem-service workflow | Python and R/R Markdown workflows, a layer table, and some service rasters/archives are present. | **Implemented, unverified** |
| Ecosystem-service results | This handoff did not confirm an output GeoPackage, final metrics, or rendered figures for the configured workflow. | **Unresolved** |
| Water-consumption/CAS material | Local files exist in a restricted data area. | **Existence only; do not expose or publish without permission and provenance review** |

## Analytical components

### 1. Mining-footprint comparison

The available scripts and summaries work with `Maus2022` and `Tang2023`
polygon datasets.  Area calculations should use an equal-area CRS.  Summed
polygon areas can double-count spatial overlap unless geometries are dissolved
or overlap is explicitly handled.  The current summary CSVs must not be
interpreted as a unioned mining footprint without that check.

### 2. LULC change, 2000–2020

`code/lulc_change_analysis.py` defines classes 1–14 and computes class areas,
net change, stable/changed areas, and a transition matrix.  It is configured
with EPSG:6933 for equal-area raster processing.  Exact pixel-level transition
work requires the two rasters to share the same grid, not merely compatible
CRS.  See `memory/RESEARCH_DECISIONS.md` for the memory-safe strategy.

### 3. Ecosystem-service footprinting

`code/es_footprint_analysis.py` is configured to calculate polygon-level
statistics including `{service}_mean`, `{service}_adj_sum`, and
`{service}_flag` from the service-layer table.  The currently configured list
includes nitrogen retention, sediment retention, and nature access.  The layer
table also includes coastal risk reduction.  Paths and availability must be
resolved before executing the workflow.

### 4. Mapping and comparison

The point-map workflow uses representative points so symbols remain inside
source polygons.  One code path applies Equal Earth (EPSG:8857) for
area-safe geometry operations.  Figure design and numeric claims remain
unverified until their input data, scale/legend logic, and rendered output are
inspected for the actual run.

## Known constraints and open checks

- Several scripts contain an absolute Windows base path. It must be replaced
  with a portable configuration before sharing or running elsewhere.
- The working tree contained pre-existing uncommitted changes when this memory
  system was added. Preserve them unless the owner directs otherwise.
- `data/` is ignored by Git, so a collaborator who clones the repository will
  not automatically obtain inputs or outputs.
- The repository may contain restricted or sensitive material. Share only
  appropriately cleared subsets and do not copy private filenames/contents
  into public documentation.

## What a new session should do

Read `AGENTS.md`, this file, `RESEARCH_DECISIONS.md`, and `NEXT_STEPS.md`.
Then inspect the current tree and select one small, reproducible verification
task rather than assuming the documented workflows have completed.
