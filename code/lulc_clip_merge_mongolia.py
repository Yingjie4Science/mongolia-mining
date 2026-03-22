"""
clip_merge_mongolia.py
Clips multiple .gpkg files to Mongolia boundary and merges into one output .gpkg.
"""

import os
import sys
import sqlite3
import time
import numpy as np
from pathlib import Path

# ============================================================
# STEP 1: FIX PROJ BEFORE RASTERIO IMPORT
# ============================================================

def _configure_proj():
    candidates = [
        Path(sys.prefix) / "Library" / "share" / "proj",
        Path(sys.prefix) / "share" / "proj",
    ]
    for proj_dir in candidates:
        db = proj_dir / "proj.db"
        if not db.exists():
            continue
        try:
            con   = sqlite3.connect(str(db))
            minor = int(con.execute(
                "SELECT value FROM metadata WHERE key='DATABASE.LAYOUT.VERSION.MINOR'"
            ).fetchone()[0])
            con.close()
            if minor < 5:
                continue
        except Exception:
            continue
        os.environ["PROJ_DATA"]    = str(proj_dir)
        os.environ["PROJ_LIB"]     = str(proj_dir)
        os.environ["PROJ_NETWORK"] = "OFF"
        print(f"[proj] PROJ_DATA → {proj_dir} (v{minor} ✓)")
        return True
    print("[proj] WARNING: valid proj.db not found")
    return False

_configure_proj()

import geopandas as gpd
from shapely.ops import unary_union

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR  = Path(r"D:\natcap\mongolia-mining\data\raw\_mining-viz\mining_lulc\Eastern_Asia")   # ← update this
OUTPUT_DIR = INPUT_DIR.parent / "clipped_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_GPKG = OUTPUT_DIR / "Eastern_Asia_Mongolia_clipped.gpkg"
OUTPUT_LAYER = "mongolia_clipped"

# ============================================================
# STEP 1: GET MONGOLIA BOUNDARY
# ============================================================

def get_mongolia_boundary() -> gpd.GeoDataFrame:
    """
    Downloads Mongolia boundary from Natural Earth via geopandas.
    No manual download needed.
    """
    print("Fetching Mongolia boundary...")

    world = gpd.read_file(
        gpd.datasets.get_path("naturalearth_lowres")
    )
    mongolia = world[world["name"] == "Mongolia"].copy()

    if len(mongolia) == 0:
        raise ValueError("Mongolia not found in dataset — check country name.")

    print(f"Mongolia CRS: {mongolia.crs}")
    print(f"Mongolia bounds: {mongolia.total_bounds}")
    return mongolia


# ============================================================
# STEP 2: CLIP EACH FILE TO MONGOLIA
# ============================================================

def clip_file_to_boundary(gpkg_path: Path,
                           clip_geom,
                           clip_crs,
                           file_idx: int,
                           total: int) -> gpd.GeoDataFrame | None:
    """
    Reads a single .gpkg, reprojects clip geometry to match,
    applies spatial filter then precise clip.

    Args:
        gpkg_path  (Path):        Input .gpkg file.
        clip_geom  (geometry):    Mongolia boundary geometry.
        clip_crs   (CRS):         CRS of the clip geometry.
        file_idx   (int):         Current file index for logging.
        total      (int):         Total number of files.

    Returns:
        Clipped GeoDataFrame or None if no features overlap.
    """
    t0 = time.time()
    print(f"\n[{file_idx}/{total}] Processing: {gpkg_path.name} ({gpkg_path.stat().st_size / 1e6:.0f} MB)")

    # ⚡ Read with bbox filter first — avoids loading the entire file
    # Get Mongolia bounds in WGS84 for initial bbox filter
    import pyproj
    from shapely.ops import transform as shp_transform
    import functools

    # Read just enough to get the file CRS cheaply
    sample = gpd.read_file(gpkg_path, rows=1, engine="pyogrio")
    file_crs = sample.crs

    # Reproject clip geometry to match file CRS if needed
    if file_crs != clip_crs:
        clip_gdf_reproj = gpd.GeoDataFrame(
            geometry=[clip_geom], crs=clip_crs
        ).to_crs(file_crs)
        clip_geom_match = clip_gdf_reproj.geometry.iloc[0]
    else:
        clip_geom_match = clip_geom

    # Get bounds for fast bbox pre-filter
    minx, miny, maxx, maxy = clip_geom_match.bounds

    # ⚡ OPTIMIZATION 1: bbox pre-filter — only reads features in Mongolia's bbox
    gdf = gpd.read_file(
        gpkg_path,
        bbox=(minx, miny, maxx, maxy),  # fast spatial pre-filter at read time
        engine="pyogrio"                 # ⚡ 5-10x faster than fiona
    )

    if len(gdf) == 0:
        print(f"  → No features in Mongolia bbox — skipping.")
        return None

    print(f"  → {len(gdf):,} features loaded after bbox filter ({time.time()-t0:.1f}s)")

    # ⚡ OPTIMIZATION 2: spatial index for precise overlap check
    t1 = time.time()
    hits = gdf.sindex.query(clip_geom_match, predicate="intersects")
    gdf  = gdf.iloc[hits].copy()

    if len(gdf) == 0:
        print(f"  → No features intersect Mongolia boundary — skipping.")
        return None

    print(f"  → {len(gdf):,} features after spatial index filter ({time.time()-t1:.1f}s)")

    # Precise clip
    t2 = time.time()
    clipped = gdf.clip(clip_geom_match)

    if len(clipped) == 0:
        print(f"  → No features remain after precise clip — skipping.")
        return None

    print(f"  → {len(clipped):,} features clipped in {time.time()-t2:.1f}s")
    print(f"  ✓ Done in {time.time()-t0:.1f}s total")

    return clipped


# ============================================================
# STEP 3: MERGE AND SAVE
# ============================================================

def merge_and_save(clipped_parts: list, output_path: Path, layer: str):
    """
    Concatenates all clipped GeoDataFrames and saves as one .gpkg.
    """
    print(f"\nMerging {len(clipped_parts)} clipped parts...")

    merged = gpd.GeoDataFrame(
        gpd.pd.concat(clipped_parts, ignore_index=True),
        crs=clipped_parts[0].crs
    )

    print(f"Total features merged: {len(merged):,}")

    # Remove duplicate geometries that may appear at file boundaries
    before = len(merged)
    merged = merged.drop_duplicates(subset=["geometry"])
    print(f"Duplicates removed: {before - len(merged):,}")

    merged.to_file(output_path, layer=layer, driver="GPKG", engine="pyogrio")
    print(f"✓ Saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    return merged


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    t_total = time.time()

    # 1. Get Mongolia boundary
    mongolia_gdf  = get_mongolia_boundary()
    mongolia_geom = unary_union(mongolia_gdf.geometry)
    mongolia_crs  = mongolia_gdf.crs

    # 2. Find all input files
    input_files = sorted(INPUT_DIR.glob("Eastern_Asia_part*.gpkg"))
    print(f"\nFound {len(input_files)} input files.")

    if len(input_files) == 0:
        sys.exit(f"ERROR: No .gpkg files found in {INPUT_DIR}")

    # 3. Clip each file
    clipped_parts = []

    for idx, gpkg_path in enumerate(input_files, start=1):
        try:
            result = clip_file_to_boundary(
                gpkg_path=gpkg_path,
                clip_geom=mongolia_geom,
                clip_crs=mongolia_crs,
                file_idx=idx,
                total=len(input_files)
            )
            if result is not None and len(result) > 0:
                clipped_parts.append(result)

        except Exception as e:
            print(f"ERROR on {gpkg_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 4. Merge and save
    if len(clipped_parts) == 0:
        print("\nNo features found within Mongolia across all files.")
    else:
        merged = merge_and_save(clipped_parts, OUTPUT_GPKG, OUTPUT_LAYER)
        print(f"\n====== Complete ======")
        print(f"Output    : {OUTPUT_GPKG}")
        print(f"Features  : {len(merged):,}")
        print(f"Total time: {time.time()-t_total:.1f}s")