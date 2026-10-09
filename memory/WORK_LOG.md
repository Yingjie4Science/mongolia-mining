# Work log

This is a concise continuity record, not a raw transcript.  Do not add secrets,
private data, or long machine-specific paths.  Mark execution and validation
separately.

## 2026-10-09 — portable memory system bootstrap

**Completed and verified in repository**

- Reviewed the pre-existing top-level README (it contained only the project
  title) before extending it; its pre-existing executable-mode change was not
  altered.
- Added `AGENTS.md`, `CLAUDE.md`, and this `memory/` directory.
- Inspected current local source code, selected data inventory, and area-summary
  CSVs.
- Confirmed that the working tree already had unrelated, uncommitted changes;
  they were preserved.

**Recorded, not rerun**

- The two mining GeoPackages and their area-summary CSVs exist locally.
- LULC and ecosystem-service workflows exist as code/configuration, but no
  end-to-end execution, output inspection, or scientific validation was done
  during this documentation update.

## Prior working history — undated summary from the referenced conversation

The following captures documented work history from the prior conversation. It
is useful context but not proof of a current successful run.

- Mining polygons were filtered/compared for Mongolia using `Maus2022` and
  `Tang2023`; spatial predicates such as `intersects`, `within`, and
  positive-area intersection were considered.
- A point-map approach was drafted to represent variable-sized polygons, with
  point size reflecting area and representative points preferred over centroids.
- A 2000–2020 LULC workflow was developed to calculate class composition,
  net change, transition matrices, and stable/changed area.
- The earlier combined-mask approach ran into memory errors; the planned
  replacement was polygon-wise raster masking with cumulative counts.
- Ecosystem-service footprint code was drafted around mean values,
  area-adjusted sums, and flags for nitrogen retention, sediment retention,
  and nature access.
- Visualization work considered fixed-color, size-scaled mean-value maps.
  A known issue to check is whether legend sizes use the same normalization as
  plotted symbols.
- R/R Markdown work was drafted for distributional comparisons across mining
  datasets; the suffix for area-adjusted sums must match the actual output
  schema (the current Python workflow uses `_adj_sum`).

## Logging template

```markdown
## YYYY-MM-DD — short task title

**Objective:**

**Inputs and code version:**

**What was run:**

**What was inspected/validated:**

**Outputs (relative paths):**

**Result status:** verified / preliminary / failed / not run

**Decisions or caveats:**

**Next action:**
```
