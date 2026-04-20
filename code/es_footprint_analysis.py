"""
============================================================
ECOSYSTEM SERVICES FOOTPRINTING ANALYSIS FOR MINING SITES
============================================================
Overlays mining polygons with global ecosystem service
rasters, computes zonal statistics (max, mean, count,
adj_sum), applies flag thresholds, and saves results.

Dependencies:
    pip install geopandas pygeoprocessing taskgraph rasterio
    pip install numpy pandas pyogrio
    (gdal is a dependency of pygeoprocessing)

Usage:
    Edit the USER INPUTS section below, then run:
        python es_footprint_analysis.py
============================================================
"""

from cProfile import label
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ── package imports ──────────────────────────────────────────────────────────
print("Importing packages...")
try:
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import pygeoprocessing
    import taskgraph
    print("  ✓ All packages imported successfully!")
except ImportError as e:
    print(f"\n  ✗ Missing package: {e}")
    print("\nInstall required packages:")
    print("  pip install geopandas pygeoprocessing taskgraph rasterio numpy pandas pyogrio")
    sys.exit(1)

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================
# USER INPUTS  –  edit only this section
# ============================================================

# Base data directory
BASE_DIR = Path(r'C:\Users\yingjiel\Documents\mongolia-mining')

# Mining polygon datasets to process (run one or both)
DATASETS = [
    # "Tang2023",
    "Maus2022",   # uncomment to also run Maus2022
]

# Path template for mining polygon files


# Path to the ecosystem services CSV table
#   Expected columns: es_id | es_value_path | flag_threshold
#   es_value_path can be absolute OR relative to the CSV's own directory
ES_TABLE_PATH = BASE_DIR / 'data' / 'natcap_footprint_data' / 'ecosystem_service_layer_table.csv'


# Output directory (will be created if it doesn't exist)
OUT_DIR = BASE_DIR / 'data' / 'es_footprint_outputs'

# Column name for ecosystem service IDs in the CSV
ES_ID_COL = 'es_id'

# Number of parallel workers for zonal statistics
#   -1 = use all available CPU cores
#    0 = run single-threaded (safer for debugging)
N_WORKERS = 0

# Output formats: 'gpkg' (GeoPackage), 'shp' (Shapefile), 'csv' (no geometry)
OUTPUT_FORMATS = ['gpkg', 'csv']


# ============================================================
# CORE FUNCTION  (adapted from footprint_stats)
# ============================================================

def footprint_stats(footprint_path: str,
                    es_table_path: str,
                    id_col: str = 'es_id',
                    n_workers: int = 0):
    """
    Calculate zonal statistics of ecosystem service rasters under
    mining footprint polygons.

    Args:
        footprint_path: path to the mining polygon vector (any GDAL format)
        es_table_path:  path to CSV with columns [es_id, es_value_path, flag_threshold]
        id_col:         name of the ES identifier column in the CSV
        n_workers:      taskgraph parallel workers (-1 = all cores, 0 = serial)

    Returns:
        GeoDataFrame with one row per polygon and columns:
            {es_id}_max, {es_id}_count, {es_id}_nodata_count,
            {es_id}_mean, {es_id}_flag, {es_id}_adj_sum
        for every ecosystem service in the table.
    """

    # ── taskgraph cache in a temp subfolder ──────────────────────────────────
    cache_dir = os.path.join(os.path.dirname(footprint_path), '.taskgraph_cache')
    os.makedirs(cache_dir, exist_ok=True)
    graph = taskgraph.TaskGraph(cache_dir, n_workers=n_workers)

    # ── load footprint polygons ───────────────────────────────────────────────
    logger.info(f"Loading footprint polygons: {footprint_path}")
    footprint_gdf = gpd.read_file(footprint_path, engine='pyogrio', fid_as_index=True)

    if not ((footprint_gdf.geom_type == 'Polygon') |
            (footprint_gdf.geom_type == 'MultiPolygon')).all():
        raise ValueError(
            'All geometries must be Polygon or MultiPolygon. '
            'Check your mining polygon file.'
        )
    logger.info(f"  Loaded {len(footprint_gdf)} polygons  |  CRS: {footprint_gdf.crs}")

    # ── load ES table ─────────────────────────────────────────────────────────
    es_df = pd.read_csv(es_table_path)
    logger.info(f"Loaded ES table with {len(es_df)} services: "
                f"{es_df[id_col].tolist()}")

    # validate required columns
    required_cols = {id_col, 'es_value_path', 'flag_threshold'}
    missing = required_cols - set(es_df.columns)
    if missing:
        raise ValueError(
            f"ES table is missing required columns: {missing}\n"
            f"Found columns: {es_df.columns.tolist()}"
        )

    # ── queue zonal statistics tasks ─────────────────────────────────────────
    logger.info("Queueing zonal statistics tasks...")
    es_id_to_task = {}

    for i, row in es_df.iterrows():
        es_id = row[id_col]

        # resolve raster path (absolute or relative to CSV location)
        raw_path = row['es_value_path']
        if os.path.isabs(raw_path):
            raster_path = raw_path
        else:
            raster_path = os.path.abspath(
                os.path.join(os.path.dirname(es_table_path), es_id, raw_path)
            )

        if not os.path.exists(raster_path):
            logger.warning(f"  ✗ Raster not found, skipping {es_id}: {raster_path}")
            continue

        # get pixel area for area-adjusted sum
        pixel_size = pygeoprocessing.get_raster_info(raster_path)['pixel_size']
        es_df.loc[i, 'pixel_area'] = abs(pixel_size[0] * pixel_size[1])

        logger.info(f"  Queuing: {es_id}  ({os.path.basename(raster_path)})")
        es_id_to_task[es_id] = graph.add_task(
            func=pygeoprocessing.zonal_statistics,
            args=((raster_path, 1), footprint_path),
            target_path_list=[],
            task_name=f'{es_id}_stats',
            store_result=True
        )

    graph.close()
    graph.join()
    logger.info("  ✓ All zonal statistics tasks complete")

    # ── unpack results into GeoDataFrame ────────────────────────────────────
    logger.info("Unpacking results into GeoDataFrame...")

    for _, row in es_df.iterrows():
        es_id = row[id_col]

        if es_id not in es_id_to_task:
            logger.warning(f"  Skipping {es_id} (no task — raster likely missing)")
            continue

        zonal_stats = es_id_to_task[es_id].get()

        # raw stats: max, sum, count, nodata_count
        for stat in ['max', 'sum', 'count', 'nodata_count']:
            footprint_gdf[f'{es_id}_{stat}'] = pd.Series(
                footprint_gdf.index.to_series().map(
                    lambda fid, s=stat, z=zonal_stats: z[fid][s]
                )
            )

        # mean = sum / count  (only where count > 0)
        footprint_gdf[f'{es_id}_mean'] = np.nan
        valid = footprint_gdf[f'{es_id}_count'] > 0
        footprint_gdf.loc[valid, f'{es_id}_mean'] = (
            footprint_gdf.loc[valid, f'{es_id}_sum']
            / footprint_gdf.loc[valid, f'{es_id}_count']
        )

        # drop sum (not in final output per original spec) -- might be useful to keep if we want to calculate total ES value per polygon in the future
        # footprint_gdf.drop(columns=[f'{es_id}_sum'], inplace=True)

        # flag: max > threshold
        footprint_gdf[f'{es_id}_flag'] = (
            footprint_gdf[f'{es_id}_max'] > row['flag_threshold']
        )

        # area-adjusted sum index = mean * polygon_area / pixel_area
        # calculate the area-adjusted sum, which should be interpreted as an index <-- based on `natcap/natural-capital-footprint-impact` implementation
        pixel_area = row.get('pixel_area', np.nan)
        footprint_gdf[f'{es_id}_adj_sum'] = (
            footprint_gdf[f'{es_id}_mean']
            * footprint_gdf.area
            / pixel_area
        )

        logger.info(f"  ✓ {es_id}: "
                    f"mean={footprint_gdf[f'{es_id}_mean'].mean():.4f}  "
                    f"flagged={footprint_gdf[f'{es_id}_flag'].sum()} polygons")

    return footprint_gdf


# ============================================================
# OUTPUT HELPERS
# ============================================================

def save_results(gdf: gpd.GeoDataFrame,
                 out_dir: Path,
                 label: str,
                 formats: list):
    """Save the results GeoDataFrame to one or more file formats."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        if fmt == 'gpkg':
            fp = out_dir / f'es_footprint_{label}.gpkg'
            gdf.to_file(fp, driver='GPKG')
            logger.info(f"  ✓ Saved GeoPackage: {fp}")

        elif fmt == 'shp':
            fp = out_dir / f'es_footprint_{label}.shp'
            gdf.to_file(fp)
            logger.info(f"  ✓ Saved Shapefile:  {fp}")

        elif fmt == 'csv':
            fp = out_dir / f'es_footprint_{label}.csv'
            # drop geometry for CSV
            gdf.drop(columns='geometry').to_csv(fp, index=True, encoding='utf-8')
            logger.info(f"  ✓ Saved CSV:        {fp}")

        else:
            logger.warning(f"  Unknown format '{fmt}', skipping")


def print_summary(gdf: gpd.GeoDataFrame, es_table_path: str,
                  id_col: str, label: str):
    """Print a concise summary of the results to the console."""
    es_df = pd.read_csv(es_table_path)

    print("\n" + "=" * 70)
    print(f"ECOSYSTEM SERVICES FOOTPRINT SUMMARY  –  {label}")
    print("=" * 70)
    print(f"  Total polygons analysed : {len(gdf)}")
    print(f"  Ecosystem services      : {len(es_df)}")
    print()

    for _, row in es_df.iterrows():
        es_id = row[id_col]
        mean_col  = f'{es_id}_mean'
        max_col   = f'{es_id}_max'
        flag_col  = f'{es_id}_flag'
        adj_col   = f'{es_id}_adj_sum'

        if mean_col not in gdf.columns:
            print(f"  [{es_id}]  — skipped (raster not found)")
            continue

        n_flagged = int(gdf[flag_col].sum()) if flag_col in gdf.columns else 0
        print(f"  [{es_id}]")
        print(f"    Mean value   : {gdf[mean_col].mean():.4f}")
        print(f"    Max value    : {gdf[max_col].max():.4f}")
        print(f"    Adj-sum mean : {gdf[adj_col].mean():.4f}")
        print(f"    Flagged      : {n_flagged} / {len(gdf)} polygons "
              f"({100*n_flagged/len(gdf):.1f}%)")
        print()

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("ECOSYSTEM SERVICES FOOTPRINTING ANALYSIS")
    print("=" * 70)

    # ── validate common inputs ────────────────────────────────────────────────
    if not BASE_DIR.exists():
        logger.error(f"Base directory not found: {BASE_DIR}")
        logger.error("Update BASE_DIR in the USER INPUTS section.")
        sys.exit(1)

    if not ES_TABLE_PATH.exists():
        logger.error(f"ES table not found: {ES_TABLE_PATH}")
        logger.error("Update ES_TABLE_PATH in the USER INPUTS section.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── process each dataset ──────────────────────────────────────────────────
    all_results = {}

    for label in DATASETS:
        mining_fp = BASE_DIR / 'data' / 'mining_polygons_clipped' / f"mining_polygons_{label}.gpkg"

        print(f"\n{'─' * 70}")
        print(f"  Processing dataset: {label}")
        print(f"{'─' * 70}")

        if not mining_fp.exists():
            logger.error(f"Mining polygon file not found: {mining_fp}")
            logger.error("Check MINING_FP_TEMPLATE in USER INPUTS.")
            continue

        try:
            result_gdf = footprint_stats(
                footprint_path=str(mining_fp),
                es_table_path=str(ES_TABLE_PATH),
                id_col=ES_ID_COL,
                n_workers=N_WORKERS
            )

            save_results(result_gdf, OUT_DIR, label, OUTPUT_FORMATS)
            print_summary(result_gdf, str(ES_TABLE_PATH), ES_ID_COL, label)
            all_results[label] = result_gdf

        except Exception as e:
            logger.error(f"Failed to process {label}: {e}")
            import traceback
            traceback.print_exc()

    # ── final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if all_results:
        print(f"✓ Analysis complete for {len(all_results)} dataset(s).")
        print(f"  Outputs saved to: {OUT_DIR}")
    else:
        print("✗ No datasets were processed successfully.")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
