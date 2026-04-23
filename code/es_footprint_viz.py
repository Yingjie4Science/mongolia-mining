"""
============================================================
ECOSYSTEM SERVICES FOOTPRINT – VISUALIZATION SUITE
============================================================
Reads the output GeoPackage(s) from es_footprint_analysis.py
and produces a full set of presentation-ready figures:

  Fig 01 – ES mean values per service (grouped bar, both datasets)
  Fig 02 – Flag rate per service (% polygons flagged)
  Fig 03 – Area-adjusted sum distribution (violin + box)
  Fig 04 – Correlation heatmap of ES mean values
  Fig 05 – Flag overlap matrix (how often services co-flag)
  Fig 06 – Spatial map: mean value per polygon, one map per ES
  Fig 07 – Spatial map: flag status overlay
  Fig 08 – Ranked polygon risk score (composite flag count)
  Fig 09 – Pairwise scatter matrix of ES mean values
  Fig 10 – Combined summary dashboard (1-page overview)

Usage:
    Edit USER INPUTS, then:
        python es_footprint_viz.py
============================================================
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

print("Importing packages...")
try:
    import geopandas as gpd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.ticker import FuncFormatter
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from shapely.geometry import box
    print("  ✓ All packages ready!\n")
except ImportError as e:
    print(f"  ✗ Missing package: {e}")
    print("  pip install geopandas matplotlib seaborn numpy pandas shapely")
    sys.exit(1)


# ============================================================
# USER INPUTS
# ============================================================

BASE_DIR  = Path(r'C:\Users\yingjiel\Documents\mongolia-mining')
OUT_DIR   = BASE_DIR / 'data' / 'es_footprint_outputs'   # where .gpkg files live
FIG_DIR   = BASE_DIR / 'figures' / 'es_footprint_figures'   # where figures are saved
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Datasets to visualize – must have matching es_footprint_{label}.gpkg in OUT_DIR
DATASETS = ["Maus2022"]          # add "Tang2023" to compare both

# Ecosystem service IDs  (must match es_id column in your ES table)
ES_IDS = [
    # "coastal_risk_reduction_service",
    "nitrogen_retention_service",
    "sediment_retention_service",
    "nature_access",
]

# Short display labels for plots (same order as ES_IDS)
ES_LABELS = {
    # "coastal_risk_reduction_service": "Coastal Risk\nReduction",
    "nitrogen_retention_service":     "Nitrogen\nRetention",
    "sediment_retention_service":     "Sediment\nRetention",
    "nature_access":                  "Nature\nAccess",
}

# Colour palette – one colour per ES service
ES_COLORS = {
    # "coastal_risk_reduction_service": "#2196F3",   # blue
    "nitrogen_retention_service":     "#4CAF50",   # green
    "sediment_retention_service":     "#FF9800",   # orange
    "nature_access":                  "#9C27B0",   # purple
}

# Dataset display labels
DS_LABELS = {
    # "Tang2023": "Tang et al. 2023",
    "Maus2022": "Maus et al. 2022",
}

# Figure DPI and style
DPI    = 300
STYLE  = "whitegrid"        # seaborn style
FONT_SCALE = 1.1


# ============================================================
# HELPERS
# ============================================================

def load_results(datasets: list, out_dir: Path) -> dict:
    """Load GeoPackage results into a dict keyed by dataset label."""
    gdfs = {}
    for label in datasets:
        fp = out_dir / f"es_footprint_{label}.gpkg"
        if not fp.exists():
            print(f"  ✗ File not found, skipping {label}: {fp}")
            continue
        gdf = gpd.read_file(fp)
        gdfs[label] = gdf
        print(f"  ✓ Loaded {label}: {len(gdf)} polygons")
    return gdfs


def es_cols(gdf: gpd.GeoDataFrame, es_id: str, stat: str) -> pd.Series:
    """Safely retrieve an ES stat column; returns NaN series if missing."""
    col = f"{es_id}_{stat}"
    if col in gdf.columns:
        return gdf[col]
    return pd.Series(np.nan, index=gdf.index)


def flag_rate(gdf, es_id):
    """Fraction of polygons flagged for a given ES."""
    col = f"{es_id}_flag"
    if col not in gdf.columns:
        return np.nan
    return gdf[col].mean() * 100   # percent


def risk_score(gdf: gpd.GeoDataFrame, es_ids: list) -> pd.Series:
    """Count how many ES services flag each polygon (composite risk score)."""
    flag_cols = [f"{e}_flag" for e in es_ids if f"{e}_flag" in gdf.columns]
    return gdf[flag_cols].sum(axis=1)


def save(fig, name: str):
    fp = FIG_DIR / name
    fig.savefig(fp, dpi=DPI, bbox_inches='tight')
    print(f"\t ✓ Saved: {fp.name}")
    plt.close(fig)


def short(es_id):
    return ES_LABELS.get(es_id, es_id)


# ============================================================
# FIGURE 01 – Grouped bar: mean ES value per service & dataset
# ============================================================

def fig01_mean_bar(gdfs):
    sns.set_theme(style=STYLE, font_scale=FONT_SCALE)
    n_es  = len(ES_IDS)
    n_ds  = len(gdfs)
    x     = np.arange(n_es)
    width = 0.35
    ds_colors = [
        # "#1f77b4", 
        "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(max(10, n_es * 2), 5))

    for i, (label, gdf) in enumerate(gdfs.items()):
        means = [es_cols(gdf, e, 'mean').mean() for e in ES_IDS]
        offset = (i - (n_ds - 1) / 2) * width
        bars = ax.bar(x + offset, means, width,
                      label=DS_LABELS.get(label, label),
                      color=ds_colors[i % len(ds_colors)],
                      edgecolor='white', linewidth=0.6)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([short(e) for e in ES_IDS])
    ax.set_ylabel("Mean Value across Polygons")
    ax.set_title("Fig 01 – Mean Ecosystem Service Values at Mining Sites",
                 fontweight='bold', pad=12)
    ax.legend(frameon=True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.1f}'))
    sns.despine()
    save(fig, "fig01_mean_bar.png")


# ============================================================
# FIGURE 02 – Flag rate per ES service (horizontal bar)
# ============================================================

def fig02_flag_rate(gdfs):
    sns.set_theme(style=STYLE, font_scale=FONT_SCALE)
    fig, ax = plt.subplots(figsize=(9, max(4, len(ES_IDS) * 1.1)))

    bar_h = 0.3
    y = np.arange(len(ES_IDS))
    ds_colors = ["#1f77b4", "#ff7f0e"]

    for i, (label, gdf) in enumerate(gdfs.items()):
        rates = [flag_rate(gdf, e) for e in ES_IDS]
        offset = (i - (len(gdfs) - 1) / 2) * bar_h
        bars = ax.barh(y + offset, rates, bar_h,
                       label=DS_LABELS.get(label, label),
                       color=ds_colors[i % 2], alpha=0.85,
                       edgecolor='white')
        for bar, r in zip(bars, rates):
            if not np.isnan(r):
                ax.text(r + 0.5, bar.get_y() + bar.get_height() / 2,
                        f'{r:.1f}%', va='center', fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([short(e) for e in ES_IDS])
    ax.set_xlabel("Polygons Flagged (%)")
    ax.set_xlim(0, 105)
    ax.axvline(50, color='red', lw=1, ls='--', alpha=0.5, label='50% line')
    ax.set_title("Fig 02 – Percentage of Mining Polygons Flagged per ES",
                 fontweight='bold', pad=12)
    ax.legend(frameon=True)
    sns.despine()
    save(fig, "fig02_flag_rate.png")


# ============================================================
# FIGURE 03 – Violin + box: adj_sum distribution per ES
# ============================================================

def fig03_violin(gdfs):
    sns.set_theme(style=STYLE, font_scale=FONT_SCALE)

    # build long-form dataframe
    rows = []
    for label, gdf in gdfs.items():
        for e in ES_IDS:
            vals = es_cols(gdf, e, 'adj_sum').dropna()
            for v in vals:
                rows.append({'ES': short(e), 'adj_sum': v,
                             'Dataset': DS_LABELS.get(label, label)})
    df = pd.DataFrame(rows)
    if df.empty:
        print("  ✗ Fig 03 skipped – no adj_sum data")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(ES_IDS) * 2.5), 6))
    palette = [ES_COLORS.get(e, '#888') for e in ES_IDS]

    sns.violinplot(data=df, x='ES', y='adj_sum', hue='Dataset',
                   inner='box', cut=0, linewidth=0.8,
                   palette=["#1f77b4", "#ff7f0e"], ax=ax)

    ax.set_xlabel("")
    ax.set_ylabel("Area-Adjusted ES Sum (index)")
    ax.set_title("Fig 03 – Distribution of Area-Adjusted ES Values at Mining Sites",
                 fontweight='bold', pad=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    sns.despine()
    save(fig, "fig03_violin_adjsum.png")


# ============================================================
# FIGURE 04 – Correlation heatmap of ES mean values
# ============================================================

def fig04_correlation(gdfs):
    sns.set_theme(style='white', font_scale=FONT_SCALE)

    for label, gdf in gdfs.items():
        mean_cols = [f"{e}_mean" for e in ES_IDS if f"{e}_mean" in gdf.columns]
        if len(mean_cols) < 2:
            continue

        corr = gdf[mean_cols].rename(
            columns={f"{e}_mean": short(e) for e in ES_IDS}
        ).corr()

        fig, ax = plt.subplots(figsize=(7, 6))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=-1, vmax=1, mask=mask,
                    square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.8, "label": "Pearson r"},
                    ax=ax)
        ax.set_title(f"Fig 04 – ES Mean Value Correlations\n{DS_LABELS.get(label, label)}",
                     fontweight='bold', pad=12)
        plt.tight_layout()
        save(fig, f"fig04_correlation_{label}.png")


# ============================================================
# FIGURE 05 – Flag co-occurrence matrix
# ============================================================

def fig05_flag_cooccurrence(gdfs):
    sns.set_theme(style='white', font_scale=FONT_SCALE)

    for label, gdf in gdfs.items():
        flag_cols = [f"{e}_flag" for e in ES_IDS if f"{e}_flag" in gdf.columns]
        if len(flag_cols) < 2:
            continue

        flags = gdf[flag_cols].rename(
            columns={f"{e}_flag": short(e) for e in ES_IDS}
        ).astype(int)

        # pairwise co-occurrence = dot product / n_polygons * 100
        co = flags.T.dot(flags) / len(flags) * 100

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(co, annot=True, fmt=".1f", cmap="YlOrRd",
                    vmin=0, vmax=100, square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8, "label": "Co-flag rate (%)"},
                    ax=ax)
        ax.set_title(
            f"Fig 05 – ES Flag Co-Occurrence Matrix\n"
            f"{DS_LABELS.get(label, label)}  "
            f"(diagonal = individual flag rate)",
            fontweight='bold', pad=12)
        plt.tight_layout()
        save(fig, f"fig05_flag_cooccurrence_{label}.png")


# ============================================================
# FIGURE 06 – Spatial maps: mean ES value per polygon
# ============================================================

def fig06_spatial_mean(gdfs):
    for label, gdf in gdfs.items():
        n = len(ES_IDS)
        ncols = min(2, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(8 * ncols, 6 * nrows))
        axes = np.array(axes).flatten()

        for i, es_id in enumerate(ES_IDS):
            ax = axes[i]
            col = f"{es_id}_mean"
            if col not in gdf.columns:
                ax.set_visible(False)
                continue

            vmin = gdf[col].quantile(0.02)
            vmax = gdf[col].quantile(0.98)

            gdf.plot(column=col, ax=ax, cmap='YlOrRd',
                     vmin=vmin, vmax=vmax,
                     legend=True,
                     legend_kwds={'label': 'Mean Value',
                                  'shrink': 0.7,
                                  'orientation': 'horizontal',
                                  'pad': 0.02},
                     missing_kwds={'color': '#cccccc', 'label': 'No data'},
                     edgecolor='none', linewidth=0)

            ax.set_title(f"{short(es_id)}", fontweight='bold', fontsize=11)
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            ax.tick_params(labelsize=8)

        # hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"Fig 06 – Spatial Distribution of ES Mean Values\n"
            f"{DS_LABELS.get(label, label)}",
            fontweight='bold', fontsize=13, y=1.01)
        plt.tight_layout()
        save(fig, f"fig06_spatial_mean_{label}.png")


# ============================================================
# FIGURE 07 – Spatial map: flag status per ES
# ============================================================

def fig07_spatial_flags(gdfs):
    for label, gdf in gdfs.items():
        n = len(ES_IDS)
        ncols = min(2, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(8 * ncols, 6 * nrows))
        axes = np.array(axes).flatten()

        flagged_color   = "#e74c3c"
        unflagged_color = "#95a5a6"

        for i, es_id in enumerate(ES_IDS):
            ax = axes[i]
            col = f"{es_id}_flag"
            if col not in gdf.columns:
                ax.set_visible(False)
                continue

            gdf_f  = gdf[gdf[col] == True]
            gdf_nf = gdf[gdf[col] == False]

            gdf_nf.plot(ax=ax, color=unflagged_color,
                        edgecolor='none', linewidth=0, label='Not flagged')
            gdf_f.plot(ax=ax, color=flagged_color,
                       edgecolor='none', linewidth=0, label='Flagged')

            n_flag = len(gdf_f)
            pct    = 100 * n_flag / len(gdf) if len(gdf) > 0 else 0
            ax.set_title(f"{short(es_id)}\n({n_flag} flagged, {pct:.1f}%)",
                         fontweight='bold', fontsize=11)
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            ax.tick_params(labelsize=8)

            patches = [
                mpatches.Patch(color=flagged_color,   label=f'Flagged ({pct:.1f}%)'),
                mpatches.Patch(color=unflagged_color, label='Not flagged'),
            ]
            ax.legend(handles=patches, loc='lower left', fontsize=8, frameon=True)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"Fig 07 – ES Flag Status at Mining Sites\n"
            f"{DS_LABELS.get(label, label)}",
            fontweight='bold', fontsize=13, y=1.01)
        plt.tight_layout()
        save(fig, f"fig07_spatial_flags_{label}.png")


# ============================================================
# FIGURE 08 – Composite risk score: polygon ranking
# ============================================================

def fig08_risk_score(gdfs):
    sns.set_theme(style=STYLE, font_scale=FONT_SCALE)

    for label, gdf in gdfs.items():
        scores = risk_score(gdf, ES_IDS)
        gdf = gdf.copy()
        gdf['risk_score'] = scores
        max_score = int(scores.max())

        # ── panel A: histogram of risk scores ────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        counts = scores.value_counts().sort_index()
        cmap   = plt.cm.get_cmap('RdYlGn_r', max_score + 1)
        colors = [cmap(i / max(max_score, 1)) for i in counts.index]

        axes[0].bar(counts.index, counts.values, color=colors,
                    edgecolor='white', linewidth=0.5)
        axes[0].set_xlabel("Number of ES Services Flagged (Risk Score)")
        axes[0].set_ylabel("Number of Mining Polygons")
        axes[0].set_title("Distribution of Composite Risk Scores", fontweight='bold')
        axes[0].set_xticks(range(max_score + 1))

        for x, y in zip(counts.index, counts.values):
            axes[0].text(x, y + 0.3, str(y), ha='center', fontsize=9)

        # ── panel B: spatial map of risk score ───────────────────────────────
        colors_list = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad'][:max_score + 1]
        cmap_disc = ListedColormap(colors_list)
        bounds    = np.arange(-0.5, max_score + 1.5, 1)
        norm      = BoundaryNorm(bounds, cmap_disc.N)

        gdf.plot(column='risk_score', ax=axes[1],
                 cmap=cmap_disc, norm=norm,
                 legend=True,
                 legend_kwds={
                     'label': 'ES Services Flagged',
                     'shrink': 0.7,
                     'ticks': range(max_score + 1),
                     'orientation': 'horizontal',
                     'pad': 0.02,
                 },
                 edgecolor='none', linewidth=0)
        axes[1].set_title("Spatial Distribution of Composite Risk", fontweight='bold')
        axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")

        fig.suptitle(
            f"Fig 08 – Composite ES Risk Score at Mining Sites\n"
            f"{DS_LABELS.get(label, label)}",
            fontweight='bold', fontsize=13)
        plt.tight_layout()
        save(fig, f"fig08_risk_score_{label}.png")


# ============================================================
# FIGURE 09 – Pairwise scatter matrix of ES mean values
# ============================================================

def fig09_scatter_matrix(gdfs):
    for label, gdf in gdfs.items():
        mean_cols = [f"{e}_mean" for e in ES_IDS if f"{e}_mean" in gdf.columns]
        if len(mean_cols) < 2:
            continue

        plot_df = gdf[mean_cols].rename(
            columns={f"{e}_mean": short(e) for e in ES_IDS}
        ).dropna()

        # colour by risk score
        scores = risk_score(gdf.loc[plot_df.index], ES_IDS)
        max_s  = max(int(scores.max()), 1)

        fig = plt.figure(figsize=(3 * len(mean_cols), 3 * len(mean_cols)))
        axes = pd.plotting.scatter_matrix(
            plot_df,
            figsize=(3 * len(mean_cols), 3 * len(mean_cols)),
            alpha=0.4,
            c=scores,
            cmap='RdYlGn_r',
            diagonal='kde',
            hist_kwds={'bins': 20},
        )
        plt.suptitle(
            f"Fig 09 – Pairwise ES Mean Value Scatter Matrix\n"
            f"{DS_LABELS.get(label, label)}  "
            f"(colour = composite risk score)",
            fontweight='bold', fontsize=12, y=1.01)
        plt.tight_layout()
        save(plt.gcf(), f"fig09_scatter_matrix_{label}.png")


# ============================================================
# FIGURE 10 – One-page summary dashboard
# ============================================================

def fig10_dashboard(gdfs):
    sns.set_theme(style=STYLE, font_scale=1.0)

    for label, gdf in gdfs.items():
        fig = plt.figure(figsize=(20, 14))
        gs  = gridspec.GridSpec(3, 4, figure=fig,
                                hspace=0.45, wspace=0.35)

        ds_label = DS_LABELS.get(label, label)

        # ── A: mean bar (top left, 2 cols) ───────────────────────────────────
        ax_a = fig.add_subplot(gs[0, :2])
        means = [es_cols(gdf, e, 'mean').mean() for e in ES_IDS]
        colors = [ES_COLORS.get(e, '#888') for e in ES_IDS]
        bars = ax_a.bar([short(e) for e in ES_IDS], means, color=colors,
                        edgecolor='white', linewidth=0.5)
        ax_a.bar_label(bars, fmt='%.1f', padding=2, fontsize=8)
        ax_a.set_ylabel("Mean Value")
        ax_a.set_title("Mean ES Values", fontweight='bold')
        ax_a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
        sns.despine(ax=ax_a)

        # ── B: flag rate (top right, 2 cols) ─────────────────────────────────
        ax_b = fig.add_subplot(gs[0, 2:])
        rates = [flag_rate(gdf, e) for e in ES_IDS]
        bars2 = ax_b.barh([short(e) for e in ES_IDS], rates,
                          color=colors, edgecolor='white', linewidth=0.5)
        ax_b.set_xlabel("Polygons Flagged (%)")
        ax_b.set_xlim(0, 105)
        ax_b.axvline(50, color='red', lw=1, ls='--', alpha=0.4)
        ax_b.set_title("Flag Rate per ES", fontweight='bold')
        for bar, r in zip(bars2, rates):
            if not np.isnan(r):
                ax_b.text(r + 1, bar.get_y() + bar.get_height() / 2,
                          f'{r:.1f}%', va='center', fontsize=8)
        sns.despine(ax=ax_b)

        # ── C: correlation heatmap (middle left) ─────────────────────────────
        ax_c = fig.add_subplot(gs[1, :2])
        mean_cols = [f"{e}_mean" for e in ES_IDS if f"{e}_mean" in gdf.columns]
        if len(mean_cols) >= 2:
            corr = gdf[mean_cols].rename(
                columns={f"{e}_mean": short(e) for e in ES_IDS}).corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                        vmin=-1, vmax=1, square=True, linewidths=0.5,
                        cbar_kws={"shrink": 0.7}, ax=ax_c,
                        annot_kws={"size": 8})
            ax_c.set_title("ES Correlation", fontweight='bold')

        # ── D: risk score histogram (middle right) ────────────────────────────
        ax_d = fig.add_subplot(gs[1, 2:])
        scores = risk_score(gdf, ES_IDS)
        counts = scores.value_counts().sort_index()
        risk_colors = ['#2ecc71', '#f1c40f', '#e67e22',
                       '#e74c3c', '#8e44ad'][:int(scores.max()) + 1]
        ax_d.bar(counts.index, counts.values,
                 color=[risk_colors[i] for i in counts.index],
                 edgecolor='white')
        ax_d.set_xlabel("ES Services Flagged")
        ax_d.set_ylabel("Polygon Count")
        ax_d.set_title("Composite Risk Score Distribution", fontweight='bold')
        ax_d.set_xticks(range(int(scores.max()) + 1))
        for x, y in zip(counts.index, counts.values):
            ax_d.text(x, y + 0.2, str(y), ha='center', fontsize=8)
        sns.despine(ax=ax_d)

        # ── E: spatial risk map (bottom row, full width) ──────────────────────
        ax_e = fig.add_subplot(gs[2, :])
        gdf_plot = gdf.copy()
        gdf_plot['risk_score'] = scores
        max_s = int(scores.max())
        risk_cmap = ListedColormap(
            ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad'][:max_s + 1]
        )
        bnorm = BoundaryNorm(np.arange(-0.5, max_s + 1.5, 1), risk_cmap.N)
        gdf_plot.plot(column='risk_score', ax=ax_e,
                      cmap=risk_cmap, norm=bnorm,
                      legend=True,
                      legend_kwds={
                          'label': 'ES Services Flagged',
                          'shrink': 0.5,
                          'ticks': list(range(max_s + 1)),
                          'orientation': 'horizontal',
                          'pad': 0.04,
                      },
                      edgecolor='none', linewidth=0)
        ax_e.set_title("Spatial Distribution of Composite ES Risk Score",
                       fontweight='bold')
        ax_e.set_xlabel("Longitude"); ax_e.set_ylabel("Latitude")

        # overall title
        fig.suptitle(
            f"Ecosystem Services Footprint Dashboard  –  {ds_label}",
            fontsize=16, fontweight='bold', y=1.01)

        save(fig, f"fig10_dashboard_{label}.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("ECOSYSTEM SERVICES FOOTPRINT – VISUALIZATION SUITE")
    print("=" * 70 + "\n")

    print("[Loading results...]")
    gdfs = load_results(DATASETS, OUT_DIR)

    if not gdfs:
        print("\n✗ No result files found. Run es_footprint_analysis.py first.")
        print(f"  Expected files in: {OUT_DIR}")
        sys.exit(1)

    print(f"\n[Generating figures → {FIG_DIR}]\n")

    steps = [
        ("Fig 01 – Mean ES values (grouped bar)",    fig01_mean_bar),
        ("Fig 02 – Flag rates (horizontal bar)",      fig02_flag_rate),
        ("Fig 03 – Adj-sum distributions (violin)",  fig03_violin),
        ("Fig 04 – ES correlation heatmap",           fig04_correlation),
        ("Fig 05 – Flag co-occurrence matrix",        fig05_flag_cooccurrence),
        ("Fig 06 – Spatial maps: mean values",        fig06_spatial_mean),
        ("Fig 07 – Spatial maps: flag status",        fig07_spatial_flags),
        ("Fig 08 – Composite risk score",             fig08_risk_score),
        ("Fig 09 – Pairwise scatter matrix",          fig09_scatter_matrix),
        ("Fig 10 – Summary dashboard",                fig10_dashboard),
    ]

    for desc, fn in steps:
        print(f"  {desc}")
        try:
            fn(gdfs)
        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"✓ All figures saved to: {FIG_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
