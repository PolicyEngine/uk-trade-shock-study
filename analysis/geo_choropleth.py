"""Constituency choropleth for the tariff stress scenarios.

Reads the cached constituency results and produces full-tariff and EPD panels
on a shared sequential loss scale. No microsimulation is run here.
"""

import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import figstyle  # noqa: E402

BOUNDARIES = ROOT / "data" / "uk_constituencies_2024.geojson"
FIGDIR = ROOT / "results" / "figures"

def constituency_map():
    """Two-panel 650-constituency choropleth (full | EPD), shared scale.

    Reads the cached per-constituency results from analysis/geo_impact.py
    (results/geo/constituency_impacts.csv, enhanced FRS 2023-24, imputed SIC,
    mean over 20 paired assignment draws, period 2025) and joins on GSS code.
    """
    import pandas as pd

    figstyle.apply_style()
    impacts = ROOT / "results" / "geo" / "constituency_impacts.csv"
    df = pd.read_csv(impacts)
    gdf = gpd.read_file(BOUNDARIES)[["GSScode", "geometry"]]
    merged = gdf.merge(df, left_on="GSScode", right_on="code", how="inner")
    missing = len(df) - len(merged)
    if missing:
        print(f"warning: {missing} constituencies did not join to a boundary")
    merged = merged.set_crs(27700, allow_override=True)

    cols = [
        ("income_change_gbp_per_person_full", "Full tariff"),
        ("income_change_gbp_per_person_epd", "EPD"),
    ]
    # Every constituency value is a loss. Plot positive loss magnitudes on a
    # sequential scale whose low end is visibly blue-grey rather than white;
    # the old diverging midpoint made small but non-zero losses disappear.
    loss_cols = []
    for col, _ in cols:
        loss_col = f"{col}_loss"
        merged[loss_col] = np.maximum(-merged[col].to_numpy(float), 0.0)
        loss_cols.append(loss_col)
    allv = np.concatenate([merged[c].to_numpy() for c in loss_cols])
    vmax = float(np.nanpercentile(allv, 95)) or float(np.nanmax(allv))
    norm = Normalize(vmin=0.0, vmax=vmax, clip=True)
    cmap = LinearSegmentedColormap.from_list(
        "visible_losses", ["#D6E4EA", "#72A9BE", "#176B87", "#083B5C"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 7.5))
    for ax, ((_, label), loss_col) in zip(axes, zip(cols, loss_cols)):
        merged.plot(column=loss_col, cmap=cmap, norm=norm, ax=ax,
                    edgecolor="#F4F7F8", linewidth=0.12)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(label)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=list(axes), orientation="horizontal",
                        fraction=0.045, pad=0.02, extend="both")
    cbar.outline.set_visible(False)
    cbar.set_label("Mean per-capita disposable-income loss (£/year)")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / "map_constituency_income_change.png"
    fig.savefig(path, dpi=figstyle.DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    if (ROOT / "results" / "geo" / "constituency_impacts.csv").exists():
        constituency_map()


if __name__ == "__main__":
    main()
