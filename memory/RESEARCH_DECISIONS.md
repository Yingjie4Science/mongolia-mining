# Research decisions and validation boundaries

Last reviewed: 2026-10-09.  Record why a choice was made, its scope, and what
would be needed to revise it.  A decision is not evidence that a workflow was
successfully executed.

## Decision register

| ID | Decision | Rationale and boundary | Status |
| --- | --- | --- | --- |
| D-01 | Use equal-area CRS for area measurements. | Geographic-coordinate area is not appropriate. Existing code uses EPSG:6933 for LULC raster work and EPSG:8857 in a point-map path. The chosen CRS must be recorded for each actual run. | **Implemented in code; run-specific application unverified** |
| D-02 | Process widely distributed mining polygons in small spatial units rather than creating one country-spanning mask. | A prior attempted country-wide workflow encountered memory allocation failures because the combined bounding box was huge. Polygon-/tile-wise processing limits peak memory. | **Documented design; full execution unverified** |
| D-03 | Aggregate only class/transition counts needed for LULC results. | Avoid retaining large national or combined raster arrays. This is a compute strategy, not a substitute for checking grids and nodata. | **Implemented conceptually; verify actual code path per run** |
| D-04 | Require aligned LULC raster grids before class-to-class transitions. | Equal CRS alone does not establish pixel correspondence. Check CRS, transform, dimensions, resolution, extent, and nodata; resample/reproject with a documented categorical method if needed. | **Required validation gate** |
| D-05 | Use `representative_point()` for polygon-symbol maps. | It guarantees the plotted point lies inside the source polygon, unlike a centroid for concave geometries. | **Implemented in `es_footprint_viz_point_maps.py`; output QA unverified** |
| D-06 | Keep mean ecosystem-service values distinct from area-adjusted sums and flags. | They represent different quantities and should not be silently substituted in maps, summaries, or comparisons. | **Implemented in footprint code; results unverified** |
| D-07 | Treat overlap explicitly in footprint-area totals. | Summing individual polygon areas can double-count overlapping mining polygons. Report whether results are feature sums or dissolved union area. | **Required interpretation gate** |
| D-08 | Do not attribute environmental change to mining from overlap alone. | A before/after or spatial overlay is descriptive; causal attribution needs a defensible comparison design, dates, and confounder treatment. | **Standing research boundary** |

## Superseded or risky approaches

### Country-wide combined masking

Prior working history records attempted allocations of roughly 7.93 GiB for a
large `uint8` array and 2.46 GiB for a boolean array.  These values describe a
past failure report, not a benchmark.  Do not recreate the approach merely
because an individual raster fits in memory.

### Reusing an output without its run record

A CSV, GeoPackage, or figure can exist without a reproducible record of its
inputs, parameters, and validation.  Before reporting it, identify the source
files, code revision, configuration, CRS/area semantics, and any manual edits.

## Decision template

Add a new entry when a method changes:

```markdown
| D-XX | Short decision | Why, alternatives considered, and scope | Verified / implemented-unverified / proposed |
```

For material decisions, also add the date, code/configuration reference,
validation evidence, and how the decision may be revisited.
