# LULC Change Analysis for Mining Sites (2000-2020)

## Overview

This analysis toolkit quantifies and visualizes Land Use/Land Cover (LULC) changes within mining polygons over a 20-year period (2000-2020). It produces comprehensive statistics, transition matrices, and visualizations.

## Features

✓ **Data Processing**
- Load LULC rasters (GeoTIFF format)
- Load mining site polygons (GeoPackage/Shapefile)
- Clip rasters to polygon boundaries
- Handle multiple datasets sequentially

✓ **Statistical Analysis**
- LULC class distributions (area and percentage)
- Transition/change matrix (from-to class changes)
- Stability metrics (stable vs. changed areas)
- Per-class change quantification

✓ **Visualizations**
- Side-by-side LULC comparison (2000 vs 2020)
- Area change bar charts
- Stability vs. change pie charts
- Transition heatmaps

✓ **Output**
- CSV tables (LULC distributions, transitions, changes)
- PNG figures (high resolution, 300 DPI)
- Comprehensive text summary report

---

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Required Packages

```bash
pip install numpy pandas matplotlib seaborn rasterio geopandas shapely
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### requirements.txt
```
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
seaborn>=0.11.0
rasterio>=1.1.0
geopandas>=0.9.0
shapely>=1.7.0
```

---

## File Structure

Expected directory structure for input data:

```
data_directory/
├── mining_polygons_clipped/
│   ├── mining_polygons_Tang2023.gpkg
│   └── mining_polygons_Maus2022.gpkg
├── LULC/
│   └── MON_LULC/
│       ├── 2000.tif
│       └── 2020.tif
└── lulc_change_outputs/          (auto-created)
    ├── Tang2023/
    │   ├── LULC_2000_Tang2023.csv
    │   ├── LULC_2020_Tang2023.csv
    │   ├── Transition_Matrix_Tang2023.csv
    │   ├── Class_Changes_Tang2023.csv
    │   ├── Summary_Report_Tang2023.txt
    │   ├── 01_LULC_Comparison_Tang2023.png
    │   ├── 02_Class_Changes_Tang2023.png
    │   ├── 03_Stability_Pie_Tang2023.png
    │   └── 04_Transition_Matrix_Tang2023.png
    └── Maus2022/
        └── [same as Tang2023]
```

---

## Usage

### Step 1: Configure Paths

Edit the script and update these variables:

```python
# Base directory
dir = Path('/path/to/your/data')  # ← UPDATE THIS

# Dataset labels (can have multiple)
data_label = "Tang2023"
# data_label = "Maus2022"

# File paths (relative to dir)
mining_fp = dir / 'mining_polygons_clipped' / (f"mining_polygons_{data_label}.gpkg")
lulc_2000_fp = dir / 'LULC/MON_LULC' / "2000.tif"
lulc_2020_fp = dir / 'LULC/MON_LULC' / "2020.tif"
```

### Step 2: Customize LULC Classes (Optional)

If your LULC scheme differs, update the dictionary:

```python
LULC_CLASSES = {
    1: "Your Class Name",
    2: "Another Class",
    ...
}
```

### Step 3: Run the Script

```bash
python lulc_change_analysis.py
```

Or run from Python:

```python
from lulc_change_analysis import analyze_lulc_change

result = analyze_lulc_change("Tang2023")
```

### Step 4: Run Multiple Datasets

To analyze multiple datasets sequentially:

```python
datasets_to_analyze = ["Tang2023", "Maus2022"]

for dataset in datasets_to_analyze:
    analyze_lulc_change(dataset)
```

---

## Output Files Explained

### CSV Tables

#### `LULC_2000_{label}.csv`
LULC class distribution in 2000:
- **Class_ID**: Numeric class identifier
- **Class_Name**: Human-readable name
- **Pixel_Count**: Number of pixels
- **Area_m2**: Area in square meters
- **Area_km2**: Area in square kilometers
- **Percentage**: Percentage of total area

#### `LULC_2020_{label}.csv`
Same structure as 2000 data.

#### `Transition_Matrix_{label}.csv`
Class transitions from 2000 to 2020:
- **From_Class**, **From_Name**: Source land use class
- **To_Class**, **To_Name**: Target land use class
- **Pixel_Count**: Number of pixels that transitioned
- **Area_km2**: Area of transition
- **Change_Type**: "Stable" (same class) or "Changed" (different class)

#### `Class_Changes_{label}.csv`
Summary of per-class changes:
- **Class_ID**, **Class_Name**
- **Area_2000_km2**, **Area_2020_km2**
- **Change_km2**: Absolute change in area
- **Change_pct**: Percentage change

### Visualizations (PNG)

#### `01_LULC_Comparison_{label}.png`
Side-by-side horizontal bar charts showing:
- Left: 2000 LULC distribution (%)
- Right: 2020 LULC distribution (%)

#### `02_Class_Changes_{label}.png`
Horizontal bar chart of area changes by class:
- Green bars: classes that increased
- Red bars: classes that decreased

#### `03_Stability_Pie_{label}.png`
Pie chart showing:
- Stable area (unchanged LULC, 2000-2020)
- Changed area (transitioned to different class)

#### `04_Transition_Matrix_{label}.png`
Heatmap of LULC transitions:
- Rows: 2000 classes
- Columns: 2020 classes
- Values: transition area (km²)
- Diagonal: stable classes

### `Summary_Report_{label}.txt`
Comprehensive text report including:
1. Overview statistics (total, stable, changed areas)
2. Full LULC distributions (2000 and 2020)
3. Per-class changes
4. Top 10 transition types
5. Class-specific summaries
6. Key findings and trends

---

## Interpretation Guide

### Key Metrics

**Stability Percentage**
- % of mining area with unchanged land use
- High = stable mining/land use conditions
- Low = dynamic transitions

**Change Percentage**
- % of mining area with changed land use
- Indicates land use intensity or reclamation

**Transition Matrix**
- Diagonal values = stable areas
- Off-diagonal = transitions
- Heatmap intensity = transition area magnitude

### Common Patterns

**Forest Loss → Barren/Built-up**: Mining expansion or deforestation

**Forest → Forest Recovery**: Forest regrowth or reforestation

**Barren/Desert → Vegetation**: Mine reclamation or natural recovery

**Water → Other**: Water feature drainage or sedimentation

---

## Troubleshooting

### Error: "No module named 'rasterio'"
```bash
pip install rasterio
```

### Error: "File not found"
- Verify file paths are correct
- Check that CRS (coordinate system) matches
- Ensure raster and vector files are in same CRS

### Error: "KeyError in LULC_CLASSES"
- Your LULC data contains class IDs not in the dictionary
- Add missing classes to `LULC_CLASSES` or update data

### Large file processing (slow)
- Consider clipping rasters to bounding box first
- Process one dataset at a time
- Increase available RAM

---

## Customization

### Modify Plot Styling

Edit the plotting functions to change colors, fonts, etc.:

```python
# Example: Change bar chart colors
axes[0].barh(df['Class_Name'], df['Percentage'], color='#YOUR_HEX_COLOR')
```

### Add Additional Metrics

Extend `calculate_change_metrics()` to compute custom statistics:

```python
# Example: Calculate conversion rates
conversion_rate = changed_area / total_area
```

### Export to Different Formats

Change output format in export section:

```python
# Export to Excel instead of CSV
lulc_2000_df.to_excel(dataset_out_dir / f"LULC_2000_{data_label}.xlsx", index=False)
```

---

## Data Requirements

### Raster (LULC)
- **Format**: GeoTIFF recommended
- **CRS**: Must be geographic or projected
- **Values**: Integer pixel values matching LULC_CLASSES keys
- **NoData**: 0 or specify in code
- **Resolution**: Consistent between 2000 and 2020

### Vector (Mining Polygons)
- **Format**: GeoPackage (.gpkg) or Shapefile (.shp)
- **CRS**: Must match raster CRS (auto-reproject if needed)
- **Geometry**: Polygon type
- **Required attributes**: None (script works on geometry alone)

---

## References

**LULC Classification Systems**
- ESA CCI Land Cover
- MODIS IGBP
- Copernicus Global Land Cover
- National LULC schemes

**Key Publications**
- Land cover change detection methods
- Mining impact assessment
- Time series analysis in GIS

---

## License & Citation

If you use this script in research, please cite the accompanying documentation.

---

## Contact & Support

For issues or questions:
1. Check troubleshooting section above
2. Review error messages carefully
3. Verify file paths and data formats
4. Check output directory exists with write permissions

---

**Last Updated**: 2024
**Version**: 1.0
