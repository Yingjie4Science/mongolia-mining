# LULC Change Analysis - Quick Start Guide

## 5-Minute Setup

### Step 1: Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn rasterio geopandas shapely
```

### Step 2: Update File Paths
Edit `lulc_change_analysis.py` and set:

```python
# Line ~70: Update this to your data directory
dir = Path('/Users/yourname/data')  # macOS example
# dir = Path('C:/Users/yourname/data')  # Windows example
# dir = Path('/home/username/data')  # Linux example
```

Verify your directory structure matches:
```
your_data_folder/
├── mining_polygons_clipped/
│   ├── mining_polygons_Tang2023.gpkg
│   └── mining_polygons_Maus2022.gpkg
├── LULC/MON_LULC/
│   ├── 2000.tif
│   └── 2020.tif
```

### Step 3: Run Analysis
```bash
python lulc_change_analysis.py
```

Results appear in: `your_data_folder/lulc_change_outputs/`

---

## Configuration Examples

### Example 1: Basic Single Dataset
```python
# In lulc_change_analysis.py, near end of file:

datasets_to_analyze = ["Tang2023"]

for dataset in datasets_to_analyze:
    result = analyze_lulc_change(dataset)
```

### Example 2: Multiple Datasets
```python
datasets_to_analyze = ["Tang2023", "Maus2022"]

for dataset in datasets_to_analyze:
    result = analyze_lulc_change(dataset)
```

### Example 3: Custom LULC Classes
```python
# If your LULC scheme is different, update this dict:

LULC_CLASSES = {
    1: "Primary Forest",
    2: "Secondary Forest",
    3: "Agricultural Land",
    4: "Urban/Built-up",
    5: "Water Bodies",
    6: "Bare Soil",
    7: "Mining Areas",
    8: "Wetlands",
    9: "Other"
}
```

### Example 4: Different Directory Structure
```python
# If your files are organized differently:

# Option A: Separate directories for each dataset
lulc_2000_fp = dir / 'Tang2023' / "LULC_2000.tif"
lulc_2020_fp = dir / 'Tang2023' / "LULC_2020.tif"
mining_fp = dir / 'Tang2023' / "mining_polygons.gpkg"

# Option B: All in one flat directory
mining_fp = dir / f"mining_polygons_{data_label}.gpkg"
lulc_2000_fp = dir / f"LULC_2000_{data_label}.tif"
lulc_2020_fp = dir / f"LULC_2020_{data_label}.tif"
```

---

## Understanding the Output

### CSV Files
Open in Excel/Sheets for inspection:

**LULC_2000_Tang2023.csv:**
```
Class_ID,Class_Name,Pixel_Count,Area_m2,Area_km2,Percentage
1,Forest,500000,50000000,50.00,45.3
2,Shrub,300000,30000000,30.00,27.2
...
```

**Transition_Matrix_Tang2023.csv:**
Shows all class-to-class conversions:
```
From_Class,From_Name,To_Class,To_Name,Pixel_Count,Area_km2,Change_Type
1,Forest,1,Forest,450000,45.00,Stable
1,Forest,10,Built-up,50000,5.00,Changed
...
```

### Summary Report
Read in any text editor:
- Overview metrics
- Detailed class distributions
- Top transitions
- Key findings

### Figures
Open PNG files in any image viewer:
- `01_LULC_Comparison`: Side-by-side LULC distributions
- `02_Class_Changes`: Area gains/losses by class
- `03_Stability_Pie`: Stable vs changed proportions
- `04_Transition_Matrix`: Heat map of conversions

---

## Common Questions

**Q: What if my LULC data has different years?**
A: Edit file paths:
```python
lulc_1995_fp = dir / 'LULC/MON_LULC' / "1995.tif"
lulc_2015_fp = dir / 'LULC/MON_LULC' / "2015.tif"
```
Then modify the analysis function to use these variables instead.

**Q: Can I run for just one mining site instead of all?**
A: Filter polygons before clipping:
```python
# In analyze_lulc_change function:
mining_gdf = mining_gdf[mining_gdf['site_name'] == 'My Site']
```

**Q: How do I export to Excel instead of CSV?**
A: Add after CSV export:
```python
# In the EXPORT section
lulc_2000_df.to_excel(dataset_out_dir / f"LULC_2000_{data_label}.xlsx", index=False)
```

**Q: What if files are very large (slow processing)?**
A: 
1. Subset polygons to test first:
```python
mining_gdf = mining_gdf.head(5)  # Test with 5 polygons
```
2. Pre-clip rasters to bounding box of mining areas
3. Increase available RAM or use more powerful machine

**Q: How do I compare results across datasets?**
A: Use utilities script:
```python
from lulc_analysis_utilities import LULCAnalysisUtilities

utils = LULCAnalysisUtilities()
comparison = utils.compare_datasets(results)
print(comparison)
```

**Q: Can I create custom visualizations?**
A: Yes! Use the data in the CSV files with matplotlib/seaborn:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('LULC_2020_Tang2023.csv')
plt.figure(figsize=(10, 6))
plt.pie(df['Area_km2'], labels=df['Class_Name'], autopct='%1.1f%%')
plt.savefig('custom_pie.png', dpi=300)
```

---

## Troubleshooting Checklist

- [ ] Verified data directory path is correct
- [ ] Confirmed file paths exist in that directory
- [ ] Checked file names match exactly (case-sensitive on Linux/Mac)
- [ ] Verified raster and vector files have compatible CRS
- [ ] Confirmed LULC class IDs in LULC_CLASSES dict match data
- [ ] Ensured output directory has write permissions
- [ ] Tested with one dataset before running all

---

## Performance Tips

**For Large Datasets:**
1. **Process separately**: Don't run all datasets at once
2. **Use SSD**: Faster I/O for large files
3. **Increase tile size**: Adjust batch processing in code
4. **Subset polygons**: Test with smaller polygon sets first

**Memory Optimization:**
```python
# Delete intermediate arrays to free memory
del clipped_2000
del clipped_2020
```

**Speed Testing:**
```python
import time
start = time.time()
# ... your code ...
elapsed = time.time() - start
print(f"Processing took {elapsed:.1f} seconds")
```

---

## Next Steps After First Run

1. **Review outputs**: Check CSV tables and figures
2. **Validate results**: Do numbers make sense?
3. **Customize visualizations**: Edit colors, add titles
4. **Compare datasets**: Use utilities for multi-dataset comparison
5. **Deep dive analysis**: Examine specific transitions
6. **Create final report**: Combine results for publication/presentation

---

## Contact & Help

If you encounter issues:
1. **Read error messages carefully** - they usually indicate the problem
2. **Check file paths** - most issues are path-related
3. **Validate data** - ensure files aren't corrupted
4. **Review README.md** - comprehensive documentation
5. **Check utilities** - additional tools available

---

## Example Output Excerpt

```
======================================================================
LULC CHANGE ANALYSIS: Tang2023
======================================================================

[1/5] Loading data...
  ✓ Loaded 247 mining polygons
  ✓ Loaded LULC 2000 (shape: (4320, 5400))
  ✓ Loaded LULC 2020 (shape: (4320, 5400))
  ✓ Pixel size: 100.0 m, Area: 10000.0 m²

[2/5] Clipping rasters to mining polygons...
  ✓ Successfully clipped 247 mining polygons

[3/5] Calculating LULC statistics...
  ✓ 2000: 8 classes identified
  ✓ 2020: 9 classes identified

[4/5] Creating transition matrix...
  ✓ Identified 35 transition types

[5/5] Generating visualizations...
  ✓ Saved: 01_LULC_Comparison_Tang2023.png
  ✓ Saved: 02_Class_Changes_Tang2023.png
  ✓ Saved: 03_Stability_Pie_Tang2023.png
  ✓ Saved: 04_Transition_Matrix_Tang2023.png

[EXPORT] Writing tables to CSV...
  ✓ Saved: LULC_2000_Tang2023.csv
  ✓ Saved: LULC_2020_Tang2023.csv
  ✓ Saved: Transition_Matrix_Tang2023.csv
  ✓ Saved: Class_Changes_Tang2023.csv

[REPORT] Generating summary statistics...
  ✓ Saved: Summary_Report_Tang2023.txt

======================================================================
✓ Analysis complete! Results saved to: /path/to/lulc_change_outputs/Tang2023
======================================================================
```

---

## Version & Updates

**Current Version:** 1.0
**Last Updated:** 2024
**Python:** 3.7+

To check for updates, review the main script header for version number.

---

**Happy analyzing! 🎉**
