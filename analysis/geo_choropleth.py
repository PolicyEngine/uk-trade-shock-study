"""Choropleth maps of the tariff shock: constituency (650 seats) and ITL1 regions.

Reads the cached scenario results (results/full_tariff_displacement.json and
results/epd_displacement.json, Monte-Carlo mean across draws), joins the
per-region mean disposable-income change per person onto ITL1 region polygons, and draws a
two-panel choropleth pair in the house style: full-tariff displacement on the
left, EPD (effective protection deflated) displacement on the right, on a
shared diverging scale (grey losses, blue gains, white at zero).

Region polygons are dissolved from the 2024 Westminster constituency
boundaries (data/uk_constituencies_2024.geojson, copied from the same local
source the AI-study template uses): English seats carry a CTR_REG government
office region, Scottish/Welsh/Northern Irish seats carry only Country, which
is the region itself.

No microsimulation is run here -- this is presentation only, off cached JSON.
"""

import json
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import figstyle  # noqa: E402

BOUNDARIES = ROOT / "data" / "uk_constituencies_2024.geojson"
FIGDIR = ROOT / "results" / "figures"

# Map the results-JSON region keys onto the boundary file's region labels.
# English constituencies carry CTR_REG; Scotland/Wales/NI carry Country only.
REGION_KEYS = {
    "NORTH_EAST": "North East",
    "NORTH_WEST": "North West",
    "YORKSHIRE": "Yorkshire and the Humber",
    "EAST_MIDLANDS": "East Midlands",
    "WEST_MIDLANDS": "West Midlands",
    "EAST_OF_ENGLAND": "East of England",
    "LONDON": "Greater London",
    "SOUTH_EAST": "South East",
    "SOUTH_WEST": "South West",
    "SCOTLAND": "Scotland",
    "WALES": "Wales",
    "NORTHERN_IRELAND": "Northern Ireland",
}

PANELS = [
    ("full_tariff_displacement.json", "Full tariff"),
    ("epd_displacement.json", "EPD"),
]


def load_regions():
    gdf = gpd.read_file(BOUNDARIES)[["CTR_REG", "Country", "geometry"]]
    # English seats: government office region; devolved nations: the country.
    gdf["region_name"] = np.where(
        gdf["CTR_REG"].fillna("").str.len() > 0, gdf["CTR_REG"], gdf["Country"]
    )
    # Repair invalid constituency polygons before the union, else GEOS raises
    # a side-location conflict during dissolve.
    gdf["geometry"] = gdf["geometry"].make_valid()
    regions = gdf[["region_name", "geometry"]].dissolve(by="region_name").reset_index()
    # The GeoJSON is mislabelled EPSG:4326 but its coordinates are already
    # British National Grid eastings/northings; assign the true CRS (no reproject).
    return regions.set_crs(27700, allow_override=True)


def region_values(path):
    """Monte-Carlo mean of the per-region income change across all draws."""
    data = json.loads(Path(path).read_text())
    per_draw = [d["region_income_change"] for d in data["draws"]]
    keys = set().union(*per_draw)
    vals = {k: float(np.mean([d.get(k, 0.0) for d in per_draw])) for k in keys}
    missing = set(REGION_KEYS) - set(vals)
    if missing:
        print(f"warning: {path.name} missing regions {sorted(missing)}")
    return {REGION_KEYS[k]: v for k, v in vals.items() if k in REGION_KEYS}


def constituency_map():
    """Two-panel 650-constituency choropleth (full | EPD), shared scale.

    Reads the cached per-constituency results from analysis/geo_impact.py
    (results/geo/constituency_impacts.csv, enhanced FRS 2023-24, imputed SIC,
    seed-0 draw, period 2025) and joins on GSS code, mirroring the sister
    study's constituency choropleth.
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

    cols = [("income_change_pp_full", "Full tariff"), ("income_change_pp_epd", "EPD")]
    allv = np.concatenate([merged[c].to_numpy() for c, _ in cols])
    vmax = float(np.nanpercentile(np.abs(allv), 95)) or float(np.abs(allv).max())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    cmap = figstyle.DIVERGING

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 7.5))
    for ax, (col, label) in zip(axes, cols):
        merged.plot(column=col, cmap=cmap, norm=norm, ax=ax,
                    edgecolor="white", linewidth=0.1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(label)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=list(axes), orientation="horizontal",
                        fraction=0.045, pad=0.02, extend="both")
    cbar.outline.set_visible(False)
    cbar.set_label("Mean disposable-income change per person (£/year)")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / "map_constituency_income_change.png"
    fig.savefig(path, dpi=figstyle.DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    figstyle.apply_style()
    regions = load_regions()

    panel_vals = []
    for fname, _ in PANELS:
        vals = region_values(ROOT / "results" / fname)
        unmatched = set(regions["region_name"]) - set(vals)
        if unmatched:
            print(f"warning: boundary regions with no value in {fname}: {sorted(unmatched)}")
        panel_vals.append(vals)

    # Shared symmetric diverging scale across both panels (grey losses, blue
    # gains, white at zero), clipped at the 95th percentile of |value| so a
    # single extreme region does not crush the gradient.
    allv = np.array([v for vals in panel_vals for v in vals.values()])
    vmax = float(np.nanpercentile(np.abs(allv), 95)) or float(np.abs(allv).max())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    cmap = figstyle.DIVERGING

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 7.5))
    for ax, vals, (_, label) in zip(axes, panel_vals, PANELS):
        gdf = regions.copy()
        gdf["value"] = gdf["region_name"].map(vals)
        gdf.plot(column="value", cmap=cmap, norm=norm, ax=ax,
                 edgecolor="white", linewidth=0.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(label)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=list(axes), orientation="horizontal",
                        fraction=0.045, pad=0.02, extend="both")
    cbar.outline.set_visible(False)
    cbar.set_label("Mean disposable-income change per person (£/year)")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / "map_region_income_change.png"
    fig.savefig(path, dpi=figstyle.DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
    if (ROOT / "results" / "geo" / "constituency_impacts.csv").exists():
        constituency_map()
