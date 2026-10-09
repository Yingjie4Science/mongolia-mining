# Next steps

Priorities are ordered to establish a reproducible baseline before new
analysis.  None of these tasks is complete merely because a script or an
example output exists.

## Priority 1 — establish an execution-ready inventory

- [ ] Create a local-only run manifest: inputs, checksums or version labels,
  licenses/access status, CRS, dimensions/resolution, nodata, and code revision.
- [ ] Confirm whether the required LULC `2000.tif` and `2020.tif` files are
  extracted and readable; do not count the archive alone as ready input.
- [ ] Resolve every path in `ecosystem_service_layer_table.csv` and identify
  any unavailable/restricted service rasters.
- [ ] Replace hard-coded Windows paths with a documented local configuration or
  command-line argument, without committing personal paths.

## Priority 2 — validate a small LULC test

- [ ] Select a small, representative subset of polygons and record their IDs
  in a local test manifest.
- [ ] Validate both LULC rasters: class codes, CRS, transform, dimensions,
  resolution, extent, and nodata.
- [ ] If grids differ, specify and record the categorical resampling/alignment
  method before computing transitions.
- [ ] Run the polygon-wise workflow, inspect dated tables and figures, and
  compare total valid-pixel area against an independently calculated reference.
- [ ] Only then run the approved full dataset(s); retain a run log and review
  results before interpretation.

## Priority 3 — validate mining-area semantics

- [ ] Recompute feature count and summed feature area for each dataset with a
  recorded equal-area CRS.
- [ ] Compute a dissolved union area or explicitly quantify overlap, then label
  all reported totals as feature sum or union area.
- [ ] Inspect the existing point maps at publication size and verify symbol
  scales and legends against source values.

## Priority 4 — execute ecosystem-service footprinting safely

- [ ] Confirm service units, thresholds, nodata treatment, and whether each
  layer is meaningful for Mongolia before use.
- [ ] Test one service on a small polygon subset with `N_WORKERS = 0` and
  inspect schema, geometry, and values.
- [ ] Run the selected service/dataset combinations and save run-specific
  outputs outside the raw-input location.
- [ ] Verify the `_mean`, `_adj_sum`, and `_flag` interpretation in both the
  Python and R workflows before producing comparisons.

## Priority 5 — reporting and sharing

- [ ] Create a short methods note for any result intended for publication:
  input provenance, dates, CRS, overlap semantics, quality checks, and limits.
- [ ] Share only cleared code and documentation; keep restricted inputs and
  sensitive mine-level material outside public releases.
- [ ] After any verified milestone, update `WORK_LOG.md`, the decision register,
  and this checklist.
