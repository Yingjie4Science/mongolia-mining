"""
============================================================
LULC CHANGE ANALYSIS FOR MINING SITES (2000-2020)
============================================================
Analyzes Land Use/Land Cover changes within mining polygons
over a 20-year period. Generates comprehensive statistics,
visualizations, and change matrices.

Author: Generated for LULC Analysis
Date: 2024
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

try:
    import rasterio
    from rasterio.mask import mask
    import geopandas as gpd
    from shapely.geometry import box
except ImportError:
    print("Required packages not found. Install with:")
    print("pip install rasterio geopandas rasterio shapely")
    exit()

# add ymd timestamp to output files for better versioning and tracking
from datetime import datetime
ymd = datetime.now().strftime("%Y%m%d")

# ============================================================
# USER INPUTS
# ============================================================

# Base directory
root = Path(r"C:\Users\yingjiel\Documents\mongolia-mining")
dir = root / "data"

# Load mining polygons
# Run each dataset separately by changing the data_label and file path
# data_label = "Tang2023"
data_label = "Maus2022"

mining_fp = dir / 'mining_polygons_clipped' / (f"mining_polygons_{data_label}.gpkg")

# Load LULC data
lulc_2000_fp = dir / 'LULC/MON_LULC' / "2000.tif"
lulc_2020_fp = dir / 'LULC/MON_LULC' / "2020.tif"

# Set output directory
out_dir = dir / 'lulc_change_outputs'
out_dir.mkdir(parents=True, exist_ok=True)

# LULC class labels
LULC_CLASSES = {
    1: "Forest",
    2: "Shrub",
    3: "Meadow",
    4: "Real steppe",
    5: "Dry steppe",
    6: "Desert steppe",
    7: "Wetland",
    8: "Water",
    9: "Cropland",
    10: "Built-up land",
    11: "Barren land",
    12: "Desert",
    13: "Sand",
    14: "Ice"
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

# def load_lulc_raster(filepath: Path) -> Tuple[np.ndarray, dict, rasterio.DatasetReader]:
#     """
#     Load LULC raster file.
    
#     Returns:
#         data: numpy array of LULC values
#         profile: rasterio profile (metadata)
#         src: rasterio dataset object
#     """
#     src = rasterio.open(filepath)
#     data = src.read(1)
#     profile = src.profile
#     return data, profile, src

def load_lulc_raster(filepath: Path,
                     target_crs: str = "EPSG:6933"
                     ) -> Tuple[np.ndarray, dict, rasterio.DatasetReader]:
    """
    Load LULC raster, reprojecting to an equal-area CRS if the source
    is geographic (degrees).  
    target_crs defaults to EPSG:6933 (WGS 84 / NSIDC EASE-Grid 2.0) — a global equal-area projection
    whose units are metres, giving accurate pixel areas everywhere.
    """
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    import tempfile, os

    src = rasterio.open(filepath)

    if src.crs and src.crs.is_geographic:
        dst_crs = rasterio.CRS.from_string(target_crs)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(crs=dst_crs, transform=transform,
                       width=width, height=height)

        # write reprojected raster to a temp file so rasterio.mask works on it
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp.close()
        with rasterio.open(tmp.name, 'w', **profile) as dst:
            reproject(source=rasterio.band(src, 1),
                      destination=rasterio.band(dst, 1),
                      src_transform=src.transform,
                      src_crs=src.crs,
                      dst_transform=transform,
                      dst_crs=dst_crs,
                      resampling=Resampling.nearest)  # nearest for categorical LULC
        src.close()
        src     = rasterio.open(tmp.name)
        profile = src.profile
        print(f"    Reprojected → {target_crs}  (temp: {tmp.name})")
    else:
        profile = src.profile

    data = src.read(1)
    return data, profile, src




def load_mining_polygons(filepath: Path) -> gpd.GeoDataFrame:
    """Load mining site polygons from GeoPackage or shapefile."""
    gdf = gpd.read_file(filepath)
    # Ensure CRS is defined; reproject if needed
    if gdf.crs is None:
        print("Warning: No CRS defined. Assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


# def clip_raster_to_polygons(raster_data: np.ndarray, 
#                              raster_src: rasterio.DatasetReader,
#                              polygons: gpd.GeoDataFrame) -> Dict[int, np.ndarray]:
#     """
#     Clip raster data to mining polygon boundaries.
    
#     Returns:
#         dict mapping polygon index to clipped raster array
#     """
#     clipped_data = {}
    
#     for idx, row in polygons.iterrows():
#         geom = [row.geometry]
#         try:
#             # Clip raster to polygon
#             clipped, transform = mask(raster_src, geom, crop=True)
#             clipped_data[idx] = clipped[0]  # Extract first band
#         except Exception as e:
#             print(f"Warning: Could not clip polygon {idx}: {e}")
#             clipped_data[idx] = None
    
#     return clipped_data

def clip_raster_to_polygons(raster_data: np.ndarray,
                             raster_src: rasterio.DatasetReader,
                             polygons: gpd.GeoDataFrame) -> Dict[int, np.ndarray]:
    """
    Clip raster to polygon boundaries.
    Polygons are reprojected to the raster CRS on-the-fly so the
    mask always aligns correctly regardless of input CRS.
    """
    clipped_data = {}

    # reproject polygons to raster CRS if they differ
    polys = polygons
    if polygons.crs and raster_src.crs and polygons.crs != raster_src.crs:
        polys = polygons.to_crs(raster_src.crs)

    for idx, row in polys.iterrows():
        geom = [row.geometry]
        try:
            clipped, transform = mask(raster_src, geom, crop=True)
            clipped_data[idx] = clipped[0]
        except Exception as e:
            print(f"Warning: Could not clip polygon {idx}: {e}")
            clipped_data[idx] = None

    return clipped_data




def calculate_lulc_distribution(raster_data: np.ndarray, 
                                pixel_area_m2: float) -> pd.DataFrame:
    """
    Calculate area and percentage of each LULC class.
    
    Args:
        raster_data: numpy array of LULC values
        pixel_area_m2: pixel area in square meters
        
    Returns:
        DataFrame with class statistics
    """
    unique, counts = np.unique(raster_data, return_counts=True)
    
    stats_list = []
    for class_id, count in zip(unique, counts):
        if class_id == 0:  # Skip nodata/background
            continue
        
        area_m2 = count * pixel_area_m2
        area_km2 = area_m2 / 1e6
        
        stats_list.append({
            'Class_ID': int(class_id),
            'Class_Name': LULC_CLASSES.get(int(class_id), f'Unknown ({class_id})'),
            'Pixel_Count': int(count),
            'Area_m2': area_m2,
            'Area_km2': area_km2
        })
    
    df = pd.DataFrame(stats_list)
    if len(df) > 0:
        total_area = df['Area_km2'].sum()
        df['Percentage'] = (df['Area_km2'] / total_area * 100).round(2)
    
    return df


def create_transition_matrix(raster_2000: np.ndarray, 
                            raster_2020: np.ndarray,
                            pixel_area_m2: float) -> pd.DataFrame:
    """
    Create LULC transition/change matrix for 2000-2020.
    
    Returns:
        DataFrame with from/to class transitions and areas
    """
    # Flatten arrays
    data_2000_flat = raster_2000.flatten()
    data_2020_flat = raster_2020.flatten()
    
    # Create transition matrix
    transitions = []
    
    unique_2000 = np.unique(data_2000_flat)
    unique_2020 = np.unique(data_2020_flat)
    all_classes = np.unique(np.concatenate([unique_2000, unique_2020]))
    
    for class_from in all_classes:
        if class_from == 0:
            continue
        for class_to in all_classes:
            if class_to == 0:
                continue
            
            # Count pixels that transitioned
            count = np.sum((data_2000_flat == class_from) & (data_2020_flat == class_to))
            
            if count > 0:
                area_m2 = count * pixel_area_m2
                area_km2 = area_m2 / 1e6
                
                transitions.append({
                    'From_Class': int(class_from),
                    'From_Name': LULC_CLASSES.get(int(class_from), f'Unknown ({class_from})'),
                    'To_Class': int(class_to),
                    'To_Name': LULC_CLASSES.get(int(class_to), f'Unknown ({class_to})'),
                    'Pixel_Count': int(count),
                    'Area_km2': area_km2,
                    'Change_Type': 'Stable' if class_from == class_to else 'Changed'
                })
    
    return pd.DataFrame(transitions)


def calculate_change_metrics(lulc_2000_df: pd.DataFrame,
                             lulc_2020_df: pd.DataFrame,
                             transition_df: pd.DataFrame) -> Dict:
    """Calculate key change metrics and statistics."""
    
    # Merge 2000 and 2020 data
    merged = pd.merge(
        lulc_2000_df[['Class_ID', 'Class_Name', 'Area_km2']].rename(columns={'Area_km2': 'Area_2000_km2'}),
        lulc_2020_df[['Class_ID', 'Class_Name', 'Area_km2']].rename(columns={'Area_km2': 'Area_2020_km2'}),
        on='Class_ID',
        how='outer',
        suffixes=('', '_2020')
    ).fillna(0)

    # Consolidate Class_Name: use 2000 value where available, fall back to 2020 or dict
    if 'Class_Name_2020' in merged.columns:
        merged['Class_Name'] = merged['Class_Name'].replace(0, pd.NA).fillna(merged['Class_Name_2020'])
        merged.drop(columns=['Class_Name_2020'], inplace=True)
    merged['Class_Name'] = merged.apply(
        lambda r: LULC_CLASSES.get(int(r['Class_ID']), f"Unknown ({int(r['Class_ID'])})")
                  if r['Class_Name'] == 0 else r['Class_Name'],
        axis=1
    )
    
    merged['Change_km2'] = merged['Area_2020_km2'] - merged['Area_2000_km2']
    merged['Change_pct'] = (merged['Change_km2'] / (merged['Area_2000_km2'] + 1e-6) * 100).round(2)
    
    # Calculate total change statistics
    stable_area = transition_df[transition_df['Change_Type'] == 'Stable']['Area_km2'].sum()
    changed_area = transition_df[transition_df['Change_Type'] == 'Changed']['Area_km2'].sum()
    total_area = stable_area + changed_area
    
    metrics = {
        'Total_Mining_Area_km2': total_area,
        'Stable_Area_km2': stable_area,
        'Changed_Area_km2': changed_area,
        'Stability_Percentage': (stable_area / total_area * 100) if total_area > 0 else 0,
        'Change_Percentage': (changed_area / total_area * 100) if total_area > 0 else 0,
        'Class_Changes': merged
    }
    
    return metrics


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_lulc_comparison(lulc_2000_df: pd.DataFrame,
                         lulc_2020_df: pd.DataFrame,
                         output_fp: Path,
                         title: str = "LULC Distribution Comparison (2000 vs 2020)"):
    """Create side-by-side bar chart comparing LULC distributions."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data
    df_2000 = lulc_2000_df.sort_values('Percentage', ascending=False)
    df_2020 = lulc_2020_df.sort_values('Percentage', ascending=False)
    
    # Plot 2000
    axes[0].barh(df_2000['Class_Name'], df_2000['Percentage'], color='steelblue')
    axes[0].set_xlabel('Percentage (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('2000', fontsize=12, fontweight='bold')
    axes[0].set_xlim(0, max(df_2000['Percentage'].max(), df_2020['Percentage'].max()) * 1.1)
    
    # Plot 2020
    axes[1].barh(df_2020['Class_Name'], df_2020['Percentage'], color='coral')
    axes[1].set_xlabel('Percentage (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('2020', fontsize=12, fontweight='bold')
    axes[1].set_xlim(0, max(df_2000['Percentage'].max(), df_2020['Percentage'].max()) * 1.1)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_fp, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_fp}")
    plt.close()



import matplotlib.ticker as mticker

def plot_class_changes(class_changes_df: pd.DataFrame,
                       output_fp: Path,
                       title: str = "LULC Area Change by Class (2000-2020)"):
    """Create bar chart showing area changes by class."""
    

    # # --- convert km² → m² ---
    # class_changes_df = class_changes_df.copy()
    # class_changes_df["Change_m2"] = class_changes_df["Change_km2"] * 1e6


    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Sort by change
    df_sorted = class_changes_df.sort_values('Change_km2')
    
    # Color based on positive/negative change
    colors = ['#d01c8b' if x < 0 else '#4dac26' for x in df_sorted['Change_km2']]
    
    ax.barh(df_sorted['Class_Name'], df_sorted['Change_km2'], color=colors)

    # # Format x-axis labels to show in millions of m²
    # ax.xaxis.set_major_formatter(
    #     mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M')
    # )
    ax.set_xlabel('Area Change (km²)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_fp, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_fp}")
    plt.close()


def plot_stability_pie(stability_pct: float,
                       change_pct: float,
                       output_fp: Path,
                       title: str = "LULC Stability vs Change (2000-2020)"):
    """Create pie chart showing stable vs changed areas."""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sizes = [stability_pct, change_pct]
    labels = [f'Stable\n({stability_pct:.1f}%)', f'Changed\n({change_pct:.1f}%)']
    colors = ['#2ecc71', '#e74c3c']
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_fp, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_fp}")
    plt.close()


def plot_transition_heatmap(transition_df: pd.DataFrame,
                            output_fp: Path,
                            title: str = "LULC Transition Matrix (2000→2020)"):
    """Create heatmap of LULC transitions."""
    
    # Create pivot table
    pivot = transition_df.pivot_table(
        values='Area_km2',
        index='From_Name',
        columns='To_Name',
        aggfunc='sum',
        fill_value=0
    )
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Area (km²)'},
                ax=ax, linewidths=0.5)
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Land Use 2020', fontsize=11, fontweight='bold')
    ax.set_ylabel('Land Use 2000', fontsize=11, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_fp, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_fp}")
    plt.close()


# ============================================================
# MAIN ANALYSIS PIPELINE
# ============================================================

def analyze_lulc_change(data_label: str):
    """
    Main function to execute full LULC change analysis.
    """
    
    print("\n" + "="*70)
    print(f"LULC CHANGE ANALYSIS: {data_label}")
    print("="*70)
    
    # Create output subdirectory for this dataset
    dataset_out_dir = out_dir / data_label
    dataset_out_dir.mkdir(parents=True, exist_ok=True)
    
    # ============ LOAD DATA ============
    print("\n[1/5] Loading data...")
    
    try:
        mining_gdf = load_mining_polygons(mining_fp)
        print(f"  ✓ Loaded {len(mining_gdf)} mining polygons")
    except Exception as e:
        print(f"  ✗ Error loading mining polygons: {e}")
        return
    
    try:
        lulc_2000, profile_2000, src_2000 = load_lulc_raster(lulc_2000_fp)
        print(f"  ✓ Loaded LULC 2000 (shape: {lulc_2000.shape})")
    except Exception as e:
        print(f"  ✗ Error loading LULC 2000: {e}")
        return
    
    try:
        lulc_2020, profile_2020, src_2020 = load_lulc_raster(lulc_2020_fp)
        print(f"  ✓ Loaded LULC 2020 (shape: {lulc_2020.shape})")
    except Exception as e:
        print(f"  ✗ Error loading LULC 2020: {e}")
        return
    
    # Calculate pixel area (assuming square pixels)
    # pixel_size = profile_2000['transform'][0]
    # pixel_area_m2 = pixel_size ** 2
    # print(f"  ✓ Pixel size: {pixel_size} m, Area: {pixel_area_m2} m²")

    # ── Pixel area in m² — exact because raster is now in an equal-area CRS ──
    pixel_w_m     = abs(profile_2000['transform'][0])   # metres (x)
    pixel_h_m     = abs(profile_2000['transform'][4])   # metres (y)
    pixel_area_m2 = pixel_w_m * pixel_h_m
    raster_crs    = src_2000.crs
    print(f"  ✓ Raster CRS : {raster_crs.to_epsg() or raster_crs.to_string()}")
    print(f"    Pixel size : {pixel_w_m:.2f} m × {pixel_h_m:.2f} m")
    print(f"    Pixel area : {pixel_area_m2:,.1f} m²  ({pixel_area_m2 / 1e6:.6f} km²)")

    
    # ============ CLIP TO MINING SITES ============
    print("\n[2/5] Clipping rasters to mining polygons...")
    
    clipped_2000 = clip_raster_to_polygons(lulc_2000, src_2000, mining_gdf)
    clipped_2020 = clip_raster_to_polygons(lulc_2020, src_2020, mining_gdf)
    
    valid_polygons = [i for i in clipped_2000.keys() 
                      if clipped_2000[i] is not None and clipped_2020[i] is not None]
    print(f"  ✓ Successfully clipped {len(valid_polygons)} mining polygons")
    
    # ============ CALCULATE STATISTICS ============
    print("\n[3/5] Calculating LULC statistics...")
    
    # Aggregate all mining sites
    all_lulc_2000 = np.concatenate([clipped_2000[i].flatten() for i in valid_polygons])
    all_lulc_2020 = np.concatenate([clipped_2020[i].flatten() for i in valid_polygons])
    
    lulc_2000_df = calculate_lulc_distribution(all_lulc_2000, pixel_area_m2)
    lulc_2020_df = calculate_lulc_distribution(all_lulc_2020, pixel_area_m2)
    
    print(f"  ✓ 2000: {len(lulc_2000_df)} classes identified")
    print(f"  ✓ 2020: {len(lulc_2020_df)} classes identified")
    
    # ============ CREATE TRANSITION MATRIX ============
    print("\n[4/5] Creating transition matrix...")
    
    transition_df = create_transition_matrix(all_lulc_2000, all_lulc_2020, pixel_area_m2)
    change_metrics = calculate_change_metrics(lulc_2000_df, lulc_2020_df, transition_df)
    
    print(f"  ✓ Identified {len(transition_df[transition_df['Change_Type']=='Changed'])} transition types")
    
    # ============ GENERATE VISUALIZATIONS ============
    print("\n[5/5] Generating visualizations...")
    
    plot_lulc_comparison(
        lulc_2000_df, lulc_2020_df,
        dataset_out_dir / f"01_LULC_Comparison_{data_label}_{ymd}.png",
        title=f"LULC Distribution Comparison - {data_label}"
    )
    
    plot_class_changes(
        change_metrics['Class_Changes'],
        dataset_out_dir / f"02_Class_Changes_{data_label}_{ymd}.png",
        title=f"LULC Area Change by Class - {data_label}"
    )
    
    plot_stability_pie(
        change_metrics['Stability_Percentage'],
        change_metrics['Change_Percentage'],
        dataset_out_dir / f"03_Stability_Pie_{data_label}_{ymd}.png",
        title=f"LULC Stability vs Change - {data_label}"
    )
    
    plot_transition_heatmap(
        transition_df,
        dataset_out_dir / f"04_Transition_Matrix_{data_label}_{ymd}.png",
        title=f"LULC Transition Matrix - {data_label}"
    )
    
    # ============ EXPORT TABLES ============
    print("\n[EXPORT] Writing tables to CSV...")
    
    lulc_2000_df.to_csv(dataset_out_dir / f"LULC_2000_{data_label}_{ymd}.csv", index=False)
    print(f"  ✓ Saved: LULC_2000_{data_label}_{ymd}.csv")
    
    lulc_2020_df.to_csv(dataset_out_dir / f"LULC_2020_{data_label}_{ymd}.csv", index=False)
    print(f"  ✓ Saved: LULC_2020_{data_label}_{ymd}.csv")
    
    transition_df.to_csv(dataset_out_dir / f"Transition_Matrix_{data_label}_{ymd}.csv", index=False)
    print(f"  ✓ Saved: Transition_Matrix_{data_label}_{ymd}.csv")
    
    change_metrics['Class_Changes'].to_csv(
        dataset_out_dir / f"Class_Changes_{data_label}_{ymd}.csv", index=False
    )
    print(f"  ✓ Saved: Class_Changes_{data_label}_{ymd}.csv")
    
    # ============ GENERATE SUMMARY REPORT ============
    print("\n[REPORT] Generating summary statistics...")
    
    summary_report = generate_summary_report(data_label, lulc_2000_df, lulc_2020_df, 
                                           change_metrics, transition_df)
    
    report_fp = dataset_out_dir / f"Summary_Report_{data_label}_{ymd}.txt"
    with open(report_fp, 'w', encoding='utf-8') as f:
        f.write(summary_report)
    print(f"  ✓ Saved: Summary_Report_{data_label}_{ymd}.txt")
    
    print("\n" + "="*70)
    print(f"✓ Analysis complete! Results saved to: {dataset_out_dir}")
    print("="*70)
    
    # Close raster sources
    src_2000.close()
    src_2020.close()
    
    return {
        'lulc_2000': lulc_2000_df,
        'lulc_2020': lulc_2020_df,
        'transitions': transition_df,
        'metrics': change_metrics,
        'output_dir': dataset_out_dir
    }


def generate_summary_report(data_label: str,
                           lulc_2000_df: pd.DataFrame,
                           lulc_2020_df: pd.DataFrame,
                           metrics: Dict,
                           transition_df: pd.DataFrame) -> str:
    """Generate comprehensive text summary report."""
    
    report = f"""
{'='*70}
LULC CHANGE ANALYSIS SUMMARY REPORT
{'='*70}

Dataset: {data_label}
Analysis Period: 2000 - 2020 (20 years)
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*70}
1. OVERVIEW STATISTICS
{'='*70}

Total Mining Area:           {metrics['Total_Mining_Area_km2']:.2f} km²
Stable Area:                 {metrics['Stable_Area_km2']:.2f} km² ({metrics['Stability_Percentage']:.1f}%)
Changed Area:                {metrics['Changed_Area_km2']:.2f} km² ({metrics['Change_Percentage']:.1f}%)

{'='*70}
2. LULC DISTRIBUTION IN 2000
{'='*70}

{lulc_2000_df.to_string(index=False)}

{'='*70}
3. LULC DISTRIBUTION IN 2020
{'='*70}

{lulc_2020_df.to_string(index=False)}

{'='*70}
4. LULC CLASS CHANGES (2000-2020)
{'='*70}

{metrics['Class_Changes'].to_string(index=False)}

{'='*70}
5. TOP 10 TRANSITION TYPES (BY AREA)
{'='*70}

"""
    
    top_transitions = transition_df[transition_df['Change_Type'] == 'Changed'].nlargest(10, 'Area_km2')
    for idx, row in top_transitions.iterrows():
        report += f"\n{row['From_Name']} → {row['To_Name']}: {row['Area_km2']:.2f} km²"
    
    report += f"""

{'='*70}
6. CLASS-SPECIFIC CHANGES
{'='*70}

"""
    
    for class_id in sorted(metrics['Class_Changes']['Class_ID'].unique()):
        if pd.isna(class_id):
            continue
        row = metrics['Class_Changes'][metrics['Class_Changes']['Class_ID'] == class_id].iloc[0]
        class_name = LULC_CLASSES.get(int(class_id), f'Unknown ({class_id})')
        
        report += f"\n{class_name}:"
        report += f"\n  2000: {row['Area_2000_km2']:.2f} km²"
        report += f"\n  2020: {row['Area_2020_km2']:.2f} km²"
        report += f"\n  Change: {row['Change_km2']:+.2f} km² ({row['Change_pct']:+.1f}%)"
    
    report += f"""

{'='*70}
7. KEY FINDINGS
{'='*70}

• {metrics['Stability_Percentage']:.1f}% of mining areas remained unchanged (stable LULC)
• {metrics['Change_Percentage']:.1f}% of mining areas experienced LULC changes
• {len(transition_df[transition_df['Change_Type']=='Changed'])} different transition types detected

"""
    
    # Identify most significant changes
    most_increased = metrics['Class_Changes'].nlargest(3, 'Change_km2')
    most_decreased = metrics['Class_Changes'].nsmallest(3, 'Change_km2')
    
    report += "Most increased classes:\n"
    for _, row in most_increased.iterrows():
        if row['Change_km2'] > 0:
            report += f"  • {row['Class_Name']}: +{row['Change_km2']:.2f} km²\n"
    
    report += "\nMost decreased classes:\n"
    for _, row in most_decreased.iterrows():
        if row['Change_km2'] < 0:
            report += f"  • {row['Class_Name']}: {row['Change_km2']:.2f} km²\n"
    
    report += f"\n{'='*70}\nEnd of Report\n{'='*70}\n"
    
    return report


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    
    # UPDATE THIS PATH to your actual data directory
    dir = root / "data"
    
    if not dir.exists():
        print(f"Error: Directory not found: {dir}")
        print("Please update the 'dir' variable to point to your data directory")
        exit()
    
    # Run analysis for each dataset
    # datasets_to_analyze = ["Tang2023"]
    datasets_to_analyze = ["Maus2022"]
    
    results = {}
    for dataset in datasets_to_analyze:
        try:
            result = analyze_lulc_change(dataset)
            results[dataset] = result
        except Exception as e:
            print(f"✗ Error analyzing {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n\nAll analyses completed!")
