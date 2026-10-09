# Data dictionary and access boundaries

Last reviewed: 2026-10-09.  This is a compact index of observed repository
assets and expected workflow fields. It does not grant redistribution rights or
establish scientific suitability. Verify every source against its licence,
metadata, and current local copy before use.

## Access rule

`data/` is ignored by Git. Treat local data as private by default. Do not
commit, upload, or cite a restricted input in public documentation until its
provenance, licence, and sharing permission are confirmed.

## Observed data assets

| Asset / location | Intended role | Current status |
| --- | --- | --- |
| `data/mining_polygons_clipped/mining_polygons_Maus2022.gpkg` | Mining polygons for the Maus2022-labelled dataset | Present locally; schema/provenance must be checked per run |
| `data/mining_polygons_clipped/mining_polygons_Tang2023.gpkg` | Mining polygons for the Tang2023-labelled dataset | Present locally; schema/provenance must be checked per run |
| `data/mining_polygons_clipped/*_area_summary_stats.csv` | Feature-count and area-summary outputs | Present; values are feature-area summaries unless independently shown to be unioned |
| `data/ne_50m_admin_0_countries_MNG.*` | Mongolia boundary context | Present locally; verify source/version and CRS before analysis |
| `data/LULC/MON_LULC.zip` | Archive associated with 2000/2020 LULC workflow | Archive present; extracted raster availability and metadata unverified |
| `data/natcap_footprint_data/ecosystem_service_layer_table.csv` | Service IDs, paths, and flag thresholds | Present and inspected; referenced paths must be resolved before running |
| `data/natcap-datahub-data/` and service archives | Potential ecosystem-service raster sources | Partial local availability observed; completeness, units, and access conditions unverified |
| `data/cas/` and related mine/water data | Potential supporting or restricted material | Treat as restricted pending provenance and permission review |

## Mining polygon fields observed in area exports

| Dataset | Selected exported attributes | Derived fields |
| --- | --- | --- |
| Maus2022 | `ISO3_CODE`, `COUNTRY_NAME`, `AREA` | `area_m2`, `area_km2` |
| Tang2023 | `OBJECTID`, `Name`, `Shape_Le_1`, `Shape_Area` | `area_m2`, `area_km2` |

`area_m2` and `area_km2` should be regenerated from the geometry in a declared
equal-area CRS if they are used in a new analysis. Do not assume original
`AREA` or `Shape_Area` has the same units or overlap semantics.

## LULC classification used by the current Python workflow

| Code | Class |
| ---: | --- |
| 1 | Forest |
| 2 | Shrub |
| 3 | Meadow |
| 4 | Real steppe |
| 5 | Dry steppe |
| 6 | Desert steppe |
| 7 | Wetland |
| 8 | Water |
| 9 | Cropland |
| 10 | Built-up land |
| 11 | Barren land |
| 12 | Desert |
| 13 | Sand |
| 14 | Ice |

This classification is **implemented in code, not independently validated**
against the current rasters. Confirm values and nodata before processing.

## Ecosystem-service layer table schema

The inspected table contains these columns:

| Field | Meaning |
| --- | --- |
| `es_id` | Service identifier used to form output field names |
| `es_value_path` | Raster path, interpreted relative to the layer-table directory by the Python workflow when not absolute |
| `flag_threshold` | Threshold used by the footprinting code; validate units and intended direction before interpreting flags |

The current table names coastal risk reduction, nitrogen retention, sediment
retention, and nature access. The current code configuration selects nitrogen
retention, sediment retention, and nature access.

## Expected ecosystem-service output fields

For each `es_id`, the Python footprint workflow documents:

| Field pattern | Intended meaning | Validation needed |
| --- | --- | --- |
| `{es_id}_mean` | Mean raster value across valid cells in a footprint | Raster units, nodata treatment, and aggregation |
| `{es_id}_adj_sum` | Area-adjusted sum derived by the workflow | Pixel-area calculation and units |
| `{es_id}_flag` | Threshold-based indicator | Threshold unit, comparison direction, and policy meaning |

Never compare `_mean`, `_adj_sum`, and `_flag` as interchangeable measures.
