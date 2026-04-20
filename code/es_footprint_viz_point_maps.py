from pathlib import Path
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as mplcm
from matplotlib.lines import Line2D


# ==
# Need to switch to the base environment (3.13.0)
# ==

# ============================================================
# Inputs
# ============================================================

root = Path(r"C:\Users\yingjiel\Documents\mongolia-mining")
dir  = root / "data"

# ── Run each dataset separately by changing this label ───────────────────────
data_label   = "Maus2022"
# data_label = "Tang2023"

mining_fp            = dir / "es_footprint_outputs" / f"es_footprint_{data_label}.gpkg"
mongolia_boundary_fp = root / "data" / "ne_50m_admin_0_countries_MNG.shp"
out_dir              = root / 'figures' / 'es_footprint_figures'
out_dir.mkdir(parents=True, exist_ok=True)

# ── Services: keys match es_id values from es_footprint_analysis.py ──────────
# value = (mean_col, adj_sum_col, flag_col, display_name)
SERVICES = {
    # "coastal_risk_reduction_service": (
    #     "coastal_risk_reduction_service_mean",
    #     "coastal_risk_reduction_service_adj_sum",
    #     "coastal_risk_reduction_service_flag",
    #     "Coastal Risk Reduction",
    # ),
    "nitrogen_retention_service": (
        "nitrogen_retention_service_mean",
        "nitrogen_retention_service_adj_sum",
        "nitrogen_retention_service_flag",
        "Nitrogen Retention",
    ),
    "sediment_retention_service": (
        "sediment_retention_service_mean",
        "sediment_retention_service_adj_sum",
        "sediment_retention_service_flag",
        "Sediment Retention",
    ),
    "nature_access_service": (
        "nature_access_service_mean",
        "nature_access_service_adj_sum",
        "nature_access_service_flag",
        "Nature Access",
    ),
}

# Optional Mongolia equal-area CRS for area-safe geometry ops
EQUAL_AREA_CRS = 8857   # Equal Earth
PLOT_CRS       = 4326   # WGS84 for display

# ── Point size range (matplotlib scatter units) ───────────────────────────────
MIN_SIZE = 12
MAX_SIZE = 220

# ── Colour maps ───────────────────────────────────────────────────────────────
CMAP_MEAN = "YlOrRd"    # colour = mean ES value
FLAG_COLOR       = "#e74c3c"   # red   = flagged
NO_FLAG_COLOR    = "#95a5a6"   # grey  = not flagged


# ============================================================
# Load data
# ============================================================

print(f"\nLoading: {mining_fp.name}")
if not mining_fp.exists():
    raise FileNotFoundError(f"Footprint output not found: {mining_fp}")

mining = gpd.read_file(mining_fp)
print(f"  ✓ {len(mining)} polygons loaded  |  CRS: {mining.crs}")

boundary = None
if mongolia_boundary_fp.exists():
    boundary = gpd.read_file(mongolia_boundary_fp)
    print(f"  ✓ Boundary loaded")
else:
    print(f"  ✗ Boundary file not found, skipping: {mongolia_boundary_fp}")

# Reproject to equal-area CRS
if mining.crs is None:
    raise ValueError("Mining layer has no CRS defined.")

mining   = mining.to_crs(EQUAL_AREA_CRS)
if boundary is not None:
    boundary = boundary.to_crs(EQUAL_AREA_CRS)

# Compute polygon area in km² (equal-area projection)
mining["polygon_area_km2"] = mining.geometry.area / 1e6

# ============================================================
# Convert polygons → representative points
# representative_point() is always inside the polygon
# ============================================================

points = mining.copy()
points["geometry"] = points.geometry.representative_point()

# Reproject to plotting CRS
points   = points.to_crs(PLOT_CRS)
if boundary is not None:
    boundary = boundary.to_crs(PLOT_CRS)

print(f"  ✓ Converted to representative points  |  CRS: EPSG:{PLOT_CRS}")


# ============================================================
# Helpers
# ============================================================

def scale_values(vals, scale="sqrt"):
    import numpy as np
    vals = np.asarray(vals, dtype=float)

    if scale == "sqrt":
        return np.sqrt(vals)
    elif scale == "log1p":
        return np.log1p(vals)
    elif scale == "linear":
        return vals
    else:
        raise ValueError(f"Unsupported scale: {scale}")
    

def make_plot_sizes(
    vals,
    scale="sqrt",
    size_range=(10, 150),
    min_size=8,
    max_size=200
):
    import numpy as np

    vals = np.asarray(vals, dtype=float)
    vals = np.where(np.isnan(vals), 0, vals)

    scaled = scale_values(vals, scale)

    if np.all(scaled == 0):
        norm = np.zeros_like(scaled)
    else:
        norm = (scaled - scaled.min()) / (scaled.max() - scaled.min())

    sizes = size_range[0] + norm * (size_range[1] - size_range[0])
    sizes = np.clip(sizes, min_size, max_size)

    return sizes


def add_size_legend_scaled(
    ax,
    raw_values,
    scale="sqrt",
    size_fn=None,
    n_bins=5,
    title="Mean",
    color="#2c7fb8",
    loc="lower left"
):
    import numpy as np
    import matplotlib.pyplot as plt

    raw_values = np.asarray(raw_values, dtype=float)
    raw_values = raw_values[~np.isnan(raw_values)]

    if len(raw_values) == 0:
        return

    # --- choose representative values (quantiles = robust) ---
    qs = np.linspace(0.1, 0.9, n_bins)
    legend_vals = np.quantile(raw_values, qs)

    # --- compute corresponding sizes ---
    sizes = size_fn(legend_vals)

    # --- create legend handles ---
    handles = [
        plt.scatter([], [], s=s, color=color, alpha=0.6, edgecolors="gray", linewidths=0.2)
        for s in sizes
    ]

    # labels = [f"{v:.2g}" for v in legend_vals]
    labels = [f"{v:,.1f}" for v in legend_vals] # add commas, 1 decimal place
    # labels = [f"{v:,.0f}" for v in legend_vals]   # add commas, no decimals

    ax.legend(
        handles,
        labels,
        title=title,
        loc=loc,
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )


def add_flag_legend(ax, flag_col_present=True, loc="lower right"):
    """Add a simple flag status legend."""
    if not flag_col_present:
        return
    handles = [
        mpatches.Patch(facecolor=FLAG_COLOR,    edgecolor="gray",
                       linewidth=0.3, label="Flagged"),
        mpatches.Patch(facecolor=NO_FLAG_COLOR, edgecolor="gray",
                       linewidth=0.3, label="Not flagged"),
    ]
    ax.legend(handles=handles, loc=loc, frameon=True,
              framealpha=0.9, fontsize=8, title="Flag status",
              title_fontsize=8.5)


def annotate_map(ax, n_total, n_flagged, service_display):
    """Add site count + flag count annotation in top-left corner."""
    txt = f"Sites: {n_total:,}"
    if n_flagged is not None:
        pct = 100 * n_flagged / n_total if n_total > 0 else 0
        txt += f"\nFlagged: {n_flagged:,} ({pct:.1f}%)"
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        fontsize=10, fontweight="bold",
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=4),
    )


# ============================================================
# Plot — two maps per service:
#   Map 1: colour = mean ES value, size = adj_sum
#   Map 2: colour = flag status,   size = adj_sum
# ============================================================

scale="sqrt"

for es_id, (mean_col, sum_col, flag_col, display_name) in SERVICES.items():

    # Check required columns for mean-size map
    missing = [c for c in [mean_col] if c not in points.columns]
    if missing:
        print(f"\nSkipping '{es_id}': missing required column(s) -> {missing}")
        continue

    print(f"\nPlotting: {display_name}")

    # Keep only columns needed for this figure
    plot_gdf = points[[mean_col, "geometry"]].copy()

    # Keep only rows with valid mean values
    plot_gdf = plot_gdf[plot_gdf[mean_col].notna()].copy()

    if plot_gdf.empty:
        print("  Skipping: no valid mean values")
        continue

    # Compute marker sizes from mean values
    plot_gdf["plot_size"] = make_plot_sizes(plot_gdf[mean_col].values, scale=scale)

    # X/Y coordinates for scatter
    xs = plot_gdf.geometry.x.values
    ys = plot_gdf.geometry.y.values

    # Shared map extent from boundary or points
    if boundary is not None and not boundary.empty:
        bx1, by1, bx2, by2 = boundary.total_bounds
    else:
        bx1, by1, bx2, by2 = plot_gdf.total_bounds

    pad_x = (bx2 - bx1) * 0.03
    pad_y = (by2 - by1) * 0.03

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot boundary if available
    if boundary is not None and not boundary.empty:
        boundary.boundary.plot(ax=ax, color="#b0b0b0", linewidth=0.8)

    # Plot mean-sized points with one fixed color
    ax.scatter(
        xs, ys,
        s=plot_gdf["plot_size"].values,
        color="#d95f0e",
        alpha=0.5,
        edgecolors="gray",
        linewidths=0.2,
        zorder=3,
    )

    # Size legend based on mean values
    add_size_legend_scaled(
        ax,
        raw_values=plot_gdf[mean_col].values,
        scale=scale,
        size_fn=lambda v: make_plot_sizes(v, scale=scale),
        title=f"{display_name}\nMean",
        color="#d95f0e",
    )

    # Annotation
    annotate_map(ax, len(plot_gdf), n_flagged=None, service_display=display_name)

    # Final styling
    ax.set_xlim(bx1 - pad_x, bx2 + pad_x)
    ax.set_ylim(by1 - pad_y, by2 + pad_y)
    ax.set_title(
        f"{display_name} at Mining Sites — {data_label}\n"
        f"Size: mean value",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.set_axis_off()

    # Save
    out_fp = out_dir / f"{data_label}_{es_id}_mean_size_{scale}.png"
    fig.savefig(out_fp, dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  ✓ Saved: {out_fp.name}")

print(f"\n{'=' * 80}")
print(f"  ✓ All maps saved to: {out_dir}")
print(f"{'=' * 80}")









## ============================================================
## Plot water consumptions using the same code structure
## =============================================================
es_id = "water_consumption"
display_name = "Water Consumption"
mean_col = "annual_water_consumption_mm3"  # replace with actual column name in your data
# mean_col = "water_intensity_m3_per_ton"    


# 1. load water consumption data (similar to mining footprint data, but with water consumption values and locations)
f_water = dir / "water_consumption_cas.gpkg"
water = gpd.read_file(f_water)
water[mean_col] = water[mean_col]*1e6  # convert from million m³ to m³

print(f"  ✓ {len(water)} water consumption points loaded  |  CRS: {water.crs}")


# Reproject to plotting CRS
points   = water.to_crs(PLOT_CRS)
if boundary is not None:
    boundary = boundary.to_crs(PLOT_CRS)

print(f"  ✓ Converted to representative points  |  CRS: EPSG:{PLOT_CRS}")


# 2. preprocess data (e.g., filter for relevant locations, compute any necessary metrics
# 3. create point maps where:
#    - colour = water consumption value (e.g., total or per capita)
#    - size = maybe the same water consumption value or another relevant metric (e.g.,

print(f"\nPlotting: {display_name}")

# Keep only columns needed for this figure
plot_gdf = points[[mean_col, "geometry"]].copy()

# Keep only rows with valid mean values
plot_gdf = plot_gdf[plot_gdf[mean_col].notna()].copy()


# Compute marker sizes from mean values
plot_gdf["plot_size"] = make_plot_sizes(plot_gdf[mean_col].values, scale=scale)

# X/Y coordinates for scatter
xs = plot_gdf.geometry.x.values
ys = plot_gdf.geometry.y.values

# Shared map extent from boundary or points
if boundary is not None and not boundary.empty:
    bx1, by1, bx2, by2 = boundary.total_bounds
else:
    bx1, by1, bx2, by2 = plot_gdf.total_bounds

pad_x = (bx2 - bx1) * 0.03
pad_y = (by2 - by1) * 0.03

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Plot boundary if available
if boundary is not None and not boundary.empty:
    boundary.boundary.plot(ax=ax, color="#b0b0b0", linewidth=0.8)

# Plot mean-sized points with one fixed color
ax.scatter(
    xs, ys,
    s=plot_gdf["plot_size"].values,
    color="#d95f0e",
    alpha=0.5,
    edgecolors="gray",
    linewidths=0.2,
    zorder=3,
)

# Size legend based on mean values
add_size_legend_scaled(
    ax,
    raw_values=plot_gdf[mean_col].values,
    scale=scale,
    size_fn=lambda v: make_plot_sizes(v, scale=scale),
    title=f"{display_name}\nMean",
    color="#d95f0e",
)

# Annotation
annotate_map(ax, len(plot_gdf), n_flagged=None, service_display=display_name)

# Final styling
ax.set_xlim(bx1 - pad_x, bx2 + pad_x)
ax.set_ylim(by1 - pad_y, by2 + pad_y)
ax.set_title(
    f"{display_name} at Mining Sites — {data_label}\n"
    f"Size: mean value",
    fontsize=13, fontweight="bold", pad=10,
)
ax.set_axis_off()

# Save
out_fp = out_dir / f"{data_label}_{es_id}_mean_size_{scale}.png"
fig.savefig(out_fp, dpi=500, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"  ✓ Saved: {out_fp.name}")