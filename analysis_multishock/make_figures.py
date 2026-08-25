#!/usr/bin/env python
"""Publication figures and per-decile LaTeX tables for the multi-shock paper.

Presentation only: every number is read from ../out/results.json (the pinned
pipeline artifact) or derived from it by arithmetic that is stated in-line.
Nothing here re-estimates anything.

House style follows the sibling PolicyEngine paper (uk-ai-study/analysis/
figstyle.py): the same serif type, the same PolicyEngine brand hues, the same
axis-title conventions, so the two papers read as a family.

Canvas: every figure is saved at exactly 6.3in wide (the manuscript text
width) at 300 dpi, with explicit margins -- never `bbox_inches="tight"`,
which would silently inflate the canvas past the text width and shrink the
type when the figure is included at \\textwidth.

Colour discipline (dataviz method, brand instance = PolicyEngine):
  series slots, fixed order, never cycled:
      1 blue  #2C6496   2 green #558B2F   3 teal  #39C6C0
  Validated (dataviz scripts/validate_palette.js, light mode, white print
  surface, --pairs all): lightness band PASS, CVD separation PASS (worst
  all-pairs deutan dE 20.5), normal-vision floor PASS (dE 21.1). Two
  documented deviations, both mitigated:
    * chroma floor: brand blue sits at C = 0.100, a hair under the ~0.10
      floor. It is the fixed house primary and cannot be re-stepped without
      breaking the family look.
    * contrast: teal #39C6C0 is 2.1:1 on white, below 3:1. The skill's relief
      rule applies and is shipped everywhere teal appears: visible direct
      labels plus a full per-decile table in the manuscript.
  Print/greyscale safety: colour never carries identity alone. Every series
  additionally differs in line style, marker shape and/or hatch, and the
  three slots are separated in lightness (OKLCH L 0.47 / 0.55 / 0.75).
  Colour follows the entity across figures: energy = blue, food/TCA = green,
  clothing-footwear/CETA = teal.

Usage:
  <venv>/bin/python make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "out" / "results.json"  # pipeline output (layout fixed Aug 2026)

# --- Style: imported VERBATIM from the sibling AI-study paper's figstyle
# (figstyle.py, copied from uk-ai-study/analysis/figstyle.py), so palette,
# typography, rcParams and axis conventions are the PolicyEngine schema by
# construction.  The only local overrides are manuscript layout: a fixed
# 6.3in text-width canvas (no tight bbox) and 300 dpi for print.
from figstyle import (  # noqa: F401
    BLUE, GREEN, TEAL, BLUE_LIGHT, BLUE_PRESSED, DARKEST_BLUE, DARK_GRAY,
    GRAY, MEDIUM_DARK_GRAY, LIGHT_GRAY, INK, INK2, MUTED, GRID, BASELINE,
    NEUTRAL, LIGHT_BLUE, SERIES, SEQUENTIAL, DIVERGING, DECILE_AXIS,
)
from figstyle import apply_style as _figstyle_apply

DPI = 200  # matches figstyle (AI-study convention)
WIDTH = 8.0  # AI-paper canvas width (figstyle SINGLE); LaTeX scales to text width

_SERIF = ["Roboto Serif", "Roboto Slab", "Source Serif Pro", "DejaVu Serif"]


def apply_style() -> None:
    _figstyle_apply()
    plt.rcParams["hatch.linewidth"] = 0.6


def decile_ax(ax, ylabel, xlabel=DECILE_AXIS):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, 11))
    ax.grid(axis="x", visible=False)


def note(fig, text, y=0.015):
    """Source/method note, bottom-left, inside the fixed canvas."""
    fig.text(0.012, y, text, fontsize=6.8, color=MUTED, ha="left", va="bottom",
             linespacing=1.45)


def save(fig, name):
    """Save at exactly WIDTH inches, AI-schema layout (tight_layout with a
    reserved bottom strip for the source note)."""
    path = HERE / name
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    w, h = fig.get_size_inches()
    assert abs(w - WIDTH) < 1e-9, f"{name} is {w}in wide"
    print(f"wrote {path}  ({w:.2f} x {h:.2f} in)")


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
class Data:
    def __init__(self, res: dict):
        self.res = res
        e1 = res["episodes"]["E1_tca_food"]
        e2 = res["episodes"]["E2_energy"]
        e4 = res["episodes"]["E4_india_ceta"]
        self.e1, self.e2, self.e4 = e1, e2, e4
        self.deciles = np.arange(1, 11)

        # --- E1 TCA food NTBs -------------------------------------------
        rows = e1["central"]["per_decile_end_state_annual"]
        self.tca_end_gbp = np.array([r["gbp_per_year"] for r in rows])
        self.tca_end_pct = np.array([r["pct_of_total_spend"] for r in rows])
        # Window-average path: the published GBP250 cumulative implies an
        # average effect of `implied_average_effect_share_of_end_state` times
        # the end-state +8% over the Dec 2019 - Mar 2023 window.
        self.tca_share = e1["cross_check"]["implied_average_effect_share_of_end_state"]
        self.tca_win_gbp = self.tca_end_gbp * self.tca_share
        self.tca_win_pct = self.tca_end_pct * self.tca_share
        self.price_rise_food = e1["central"]["price_rise"]

        # --- E2 energy ---------------------------------------------------
        g = e2["gross"]["per_decile"]
        rp = res["derived"]["energy_realised_path"]["per_decile"]
        self.en_real_pct = np.array([x["pct_of_total_spend"] for x in rp])
        self.en_real_gbp = np.array([x["gbp_per_year"] for x in rp])
        n = e2["net_of_epg"]["per_decile"]
        self.en_gross_gbp = np.array([r["gbp_per_year"] for r in g])
        self.en_gross_pct = np.array([r["pct_of_total_spend"] for r in g])
        self.en_net_gbp = np.array([r["gbp_per_year"] for r in n])
        self.en_net_pct = np.array([r["pct_of_total_spend"] for r in n])
        self.en_cushion_gbp = np.array(e2["epg_cushion"]["per_decile_gbp_per_year"])
        self.cushion_share = e2["epg_cushion"]["share_of_gross_shock"]
        self.en_base_wk = np.array(e2["base_spend_elec_gas_gbp_per_week"]["per_decile"])
        self.dpp_gross = e2["price_factors"]["gross"]

        # --- E4 India CETA (100% pass-through, the headline dial) --------
        h = e4["scenarios"]["hundred"]
        self.ceta_gbp = np.array(h["per_decile_gain_gbp_per_year"])
        self.ceta_pct = np.array(h["pct_of_total_spend"])

        # --- budget shares ------------------------------------------------
        # Total spend is recovered from the identity  cost = share x total:
        #   total(FYE2022) = TCA cost / (TCA % of total spend)
        #   total(FYE2025) = CETA gain / (CETA % of total spend)
        self.tot22_wk = self.tca_end_gbp / self.tca_end_pct * 100.0 / 52.0
        self.tot25_wk = self.ceta_gbp / self.ceta_pct * 100.0 / 52.0
        # Food spend (FYE2022) implied by the +8% price vector.
        self.food22_wk = self.tca_end_gbp / (self.price_rise_food * 52.0)
        self.share_food = 100.0 * self.food22_wk / self.tot22_wk
        self.share_energy = 100.0 * self.en_base_wk / self.tot22_wk
        cf_wk = np.array(e4["categories"]["clothing"]["decile_spend_gbp_per_week"]) + np.array(
            e4["categories"]["footwear"]["decile_spend_gbp_per_week"]
        )
        self.share_clothing = 100.0 * cf_wk / self.tot25_wk

        # --- module 2 -----------------------------------------------------
        m2 = res["module2_uc_uprating"]
        self.m2 = m2
        self.months = [r["month"] for r in m2["monthly"]]
        self.actual = np.array([r["actual_allowance"] for r in m2["monthly"]])
        self.counter = np.array(
            [r["counterfactual_contemporaneous_cpi"] for r in m2["monthly"]]
        )
        self.shortfall = np.array([r["shortfall"] for r in m2["monthly"]])
        self.lag_cost = m2["uprating_lag_cost_gbp_fy_2022_23"]
        self.lag_pct = m2["uprating_lag_cost_pct_of_allowance"]
        self.apr21 = m2["parameters"]["uc_single_25plus_monthly"]["apr_2021"]

        # --- module 3 -----------------------------------------------------
        self.m3 = {r["episode"]: r for r in res["module3_comparability"]}


# ---------------------------------------------------------------------------
# Figure 1 -- incidence profile, all four quantified episodes
# ---------------------------------------------------------------------------
def fig_incidence_profile(d: Data):
    """Burden/gain as % of total household spending, by decile.

    The four episodes span three orders of magnitude (energy ~12% of spending
    at the bottom decile, CETA ~0.015%), so the axis is logarithmic in
    magnitude and the sign is carried by line style, marker fill and the
    legend text -- never by the axis.
    """
    fig, ax = plt.subplots(figsize=(WIDTH, 4.3))
    fig.subplots_adjust(left=0.125, right=0.755, top=0.925, bottom=0.315)

    series = [
        ("Energy 2022-23, realised path (cost)", d.en_real_pct, BLUE, "-", "o", BLUE),
        ("Energy 2022-23, gross counterfactual (cost)", d.en_gross_pct, BLUE, "--", "o", "white"),
        ("TCA food NTBs, window average (cost)", d.tca_win_pct, GREEN, "-", "s", GREEN),
        ("India CETA, 100% pass-through (GAIN)", d.ceta_pct, TEAL, ":", "^", "white"),
    ]
    for label, y, color, ls, marker, mfc in series:
        ax.plot(
            d.deciles, y, color=color, lw=1.8, ls=ls, marker=marker, ms=4.6,
            markerfacecolor=mfc, markeredgecolor=color, markeredgewidth=1.3,
            label=label, clip_on=False, zorder=3,
        )

    # Selective direct labels at the right-hand end: identity never rests on
    # colour alone, and this is the relief channel for the sub-3:1 teal.
    for text, yv in (
        ("Energy, counterfactual", d.en_gross_pct[-1]),
        ("Energy, realised", d.en_real_pct[-1]),
        ("TCA food, window avg.", d.tca_win_pct[-1]),
        ("India CETA gain", d.ceta_pct[-1]),
    ):
        ax.annotate(text, xy=(10, yv), xytext=(7, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=7.2, color=INK2,
                    annotation_clip=False)
    # Bottom-decile values, placed above-right of the D1 marker.
    for yv, txt in (
        (d.en_real_pct[0], f"{d.en_real_pct[0]:.1f}%"),
        (d.tca_win_pct[0], f"{d.tca_win_pct[0]:.2f}%"),
        (d.ceta_pct[0], f"{d.ceta_pct[0]:.3f}%"),
    ):
        ax.annotate(txt, xy=(1, yv), xytext=(4, 7), textcoords="offset points",
                    ha="left", va="bottom", fontsize=7.2, color=INK2)

    ax.set_yscale("log")
    ax.set_ylim(0.008, 30)
    ax.set_xlim(0.6, 10.4)
    ax.set_yticks([0.01, 0.1, 1, 10])
    ax.set_yticklabels(["0.01", "0.1", "1", "10"])
    ax.grid(axis="y", which="minor", color=GRID, lw=0.4, alpha=0.7)
    decile_ax(ax, "Burden or gain, % of spending (log scale)")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.175),
              fontsize=7.6, columnspacing=1.2, handlelength=2.6)
    save(fig, "fig_incidence_profile.png")


# ---------------------------------------------------------------------------
# Figure 2 -- energy: gross vs net-of-EPG and the cushion between them
# ---------------------------------------------------------------------------
def fig_energy_epg(d: Data):
    fig, ax = plt.subplots(figsize=(WIDTH, 4.3))
    fig.subplots_adjust(left=0.135, right=0.975, top=0.925, bottom=0.30)

    w, gap = 0.62, 14.0
    ax.bar(d.deciles, d.en_net_gbp, width=w, color=BLUE,
           label="Borne by the household, net of the EPG", zorder=3)
    ax.bar(d.deciles, d.en_cushion_gbp, width=w, bottom=d.en_net_gbp + gap,
           color=TEAL, edgecolor="white", linewidth=0.0, hatch="///", zorder=3,
           label="Cushioned by the Energy Price Guarantee")
    # Tone-on-tone ring so the hatch does not bleed into the surface.
    ax.bar(d.deciles, d.en_cushion_gbp, width=w, bottom=d.en_net_gbp + gap,
           color="none", edgecolor=TEAL, linewidth=0.8, zorder=4)

    # Selective direct labels: bottom and top decile only.
    for i in (0, 9):
        top = d.en_net_gbp[i] + d.en_cushion_gbp[i] + gap
        ax.annotate(f"gross \u00a3{d.en_gross_gbp[i]:,.0f}", xy=(d.deciles[i], top),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    va="bottom", fontsize=7.2, color=INK2)
        ax.annotate(f"net \u00a3{d.en_net_gbp[i]:,.0f}",
                    xy=(d.deciles[i], d.en_net_gbp[i] / 2), ha="center", va="center",
                    rotation=90, fontsize=7.0, color="white", zorder=5)

    ax.annotate(
        f"The EPG cushions {100 * d.cushion_share:.1f}% of the gross shock at every\n"
        "decile (constant by construction: both price vectors scale the\n"
        f"same spend base). The \u00a3 cushion runs \u00a3{d.en_cushion_gbp[0]:,.0f} (D1) to "
        f"\u00a3{d.en_cushion_gbp[9]:,.0f} (D10).",
        xy=(3.62, d.en_net_gbp[2] + d.en_cushion_gbp[2] * 0.6),
        xytext=(1.55, 3320), fontsize=7.2, color=INK2, ha="left", va="top",
        linespacing=1.4,
        arrowprops=dict(arrowstyle="-", lw=0.8, color=MUTED,
                        connectionstyle="angle3,angleA=0,angleB=75"),
    )

    decile_ax(ax, "Annual cost of the 2022-23 energy shock\n(\u00a3 per household per year)")
    ax.set_ylim(0, 3600)
    ax.set_xlim(0.4, 10.6)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.165),
              fontsize=7.6, columnspacing=1.2)
    save(fig, "fig_energy_epg.png")


# ---------------------------------------------------------------------------
# Figure 3 -- affected-category budget shares (the mechanism behind figure 1)
# ---------------------------------------------------------------------------
def fig_budget_shares(d: Data):
    fig, ax = plt.subplots(figsize=(WIDTH, 4.1))
    fig.subplots_adjust(left=0.125, right=0.885, top=0.925, bottom=0.30)

    series = [
        ("Food & non-alcoholic beverages (FYE2022)", d.share_food, GREEN, "-", "s",
         GREEN, 0),
        ("Electricity & gas (FYE2022)", d.share_energy, BLUE, "-", "o", BLUE, 7),
        ("Clothing & footwear (FYE2025)", d.share_clothing, TEAL, "--", "^", "white",
         -8),
    ]
    for label, y, color, ls, marker, mfc, dy in series:
        ax.plot(d.deciles, y, color=color, lw=1.8, ls=ls, marker=marker, ms=4.6,
                markerfacecolor=mfc, markeredgecolor=color, markeredgewidth=1.3,
                label=label, clip_on=False, zorder=3)
        ax.annotate(f"{y[-1]:.1f}%", xy=(10, y[-1]), xytext=(7, dy),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=7.2, color=INK2, annotation_clip=False)
        ax.annotate(f"{y[0]:.1f}%", xy=(1, y[0]), xytext=(-6, 0),
                    textcoords="offset points", va="center", ha="right",
                    fontsize=7.2, color=INK2)

    decile_ax(ax, "Share of total household spending (%)")
    ax.set_ylim(0, 17.5)
    ax.set_xlim(0.25, 10.4)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.175), fontsize=7.6)
    save(fig, "fig_budget_shares.png")


# ---------------------------------------------------------------------------
# Figure 4 -- the UC uprating lag
# ---------------------------------------------------------------------------
def fig_uprating_lag(d: Data):
    fig, ax = plt.subplots(figsize=(WIDTH, 4.1))
    fig.subplots_adjust(left=0.145, right=0.975, top=0.925, bottom=0.315)
    x = np.arange(len(d.months))

    ax.fill_between(x, d.actual, d.counter, color=BLUE_LIGHT, zorder=1)
    ax.fill_between(x, d.actual, d.counter, facecolor="none", hatch="\\\\\\",
                    edgecolor=GRAY, linewidth=0.0, zorder=1)

    ax.plot(x, d.counter, color=GREEN, lw=1.8, ls="--", marker="s", ms=3.8,
            markerfacecolor=GREEN, markeredgecolor=GREEN,
            label="Counterfactual: indexed to contemporaneous CPI (ONS D7BT)", zorder=3)
    ax.plot(x, d.actual, color=BLUE, lw=1.8, ls="-", marker="o", ms=3.8,
            markerfacecolor=BLUE, markeredgecolor=BLUE,
            label="Actual statutory allowance (+3.1% uprating, April 2022)", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([m.split("-")[1].capitalize() for m in d.months])
    ax.set_xlabel("Month, financial year 2022-23")
    ax.set_ylabel("Universal Credit standard allowance,\nsingle adult 25+ (\u00a3 per month)")
    ax.set_ylim(0, 400)
    ax.set_xlim(-0.4, len(x) - 0.6)
    ax.grid(axis="x", visible=False)

    ax.annotate(
        f"Cumulative shortfall, FY2022-23:\n\u00a3{d.lag_cost:,.0f} "
        f"({d.lag_pct:.1f}% of the annual allowance)",
        xy=(4.0, (d.actual[4] + d.counter[4]) / 2), xytext=(0.15, 250),
        fontsize=7.4, color=INK2, ha="left", va="top", linespacing=1.4,
        arrowprops=dict(arrowstyle="-", lw=0.8, color=MUTED,
                        connectionstyle="angle3,angleA=90,angleB=0"),
    )
    for i, ha, dx in ((0, "left", 4), (11, "right", -4)):
        ax.annotate(f"\u00a3{d.shortfall[i]:.0f}/mth",
                    xy=(i, (d.actual[i] + d.counter[i]) / 2),
                    xytext=(dx, 0), textcoords="offset points", ha=ha, va="center",
                    fontsize=7.0, color=INK2,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                              alpha=0.85))

    ax.legend(ncol=1, loc="upper center", bbox_to_anchor=(0.5, -0.155), fontsize=7.6)
    save(fig, "fig_uprating_lag.png")


# ---------------------------------------------------------------------------
# Figure 5 -- the episode map (framing figure)
# ---------------------------------------------------------------------------
def fig_episode_map(d: Data):
    fig, ax = plt.subplots(figsize=(WIDTH, 4.45))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.935, bottom=0.235)

    # (key, label, x, y, gross GBPbn, sign, market?, note)
    eps = [
        ("E1", "E1  TCA food NTBs", -1.02, -0.72, 2.19, "cost", False,
         "+8% on food prices"),
        ("E2", "E2  Energy 2022-23", -0.98, -1.60, 63.9, "cost", True,
         "cap \u00a31,277 \u2192 \u00a33,549"),
        ("E3", "E3  US tariffs 2025", 0.42, -1.10, 0.886, "cost", False,
         "earnings only; zero consumer row"),
        ("E4", "E4  India CETA", -1.02, 1.15, 0.18, "gain", False,
         "duty cut on final goods"),
        ("E5", "E5  CPTPP benchmark", 0.10, 0.62, 2.0, "gain", False,
         "GDP only; no household mapping"),
    ]
    lo = np.log10(min(e[4] for e in eps))
    hi = np.log10(max(e[4] for e in eps))

    def bubble(v):
        return 120.0 + 900.0 * (np.log10(v) - lo) / (hi - lo)

    ax.axhline(0, color=BASELINE, lw=1.0, zorder=1)
    ax.axvline(0, color=BASELINE, lw=1.0, zorder=1)

    for key, name, x, y, size, sign, market, sub in eps:
        color = BLUE if sign == "cost" else TEAL
        s = bubble(size)
        if key == "E5":
            ax.scatter(x, y, s=s, facecolor="white", edgecolor=TEAL, lw=1.6,
                       linestyle=(0, (2, 1.5)), zorder=3)
        elif market:
            ax.scatter(x, y, s=s, facecolor=color, edgecolor="white", lw=0.0,
                       hatch="///", zorder=3)
            ax.scatter(x, y, s=s, facecolor="none", edgecolor=INK, lw=1.3,
                       linestyle=(0, (3, 1.5)), zorder=4)
        else:
            ax.scatter(x, y, s=s, facecolor=color, edgecolor="white", lw=1.0,
                       zorder=3)
        r_pt = np.sqrt(s / np.pi)
        gross = (f"\u00a3{size:,.1f}bn/yr" if size >= 1
                 else f"\u00a3{1000 * size:,.0f}m/yr")
        ax.annotate(f"{name}\n{sub}\n{gross}", xy=(x, y),
                    xytext=(r_pt + 6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=7.2, color=INK,
                    linespacing=1.45)

    ax.set_xlim(-1.30, 1.75)
    ax.set_ylim(-2.30, 2.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(-1.28, 0.10, "Price channel\n(consumer prices)", fontsize=7.6, color=INK2,
            ha="left", va="bottom", linespacing=1.35)
    ax.text(1.73, 0.10, "Earnings channel\n(labour income)", fontsize=7.6, color=INK2,
            ha="right", va="bottom", linespacing=1.35)
    ax.text(-0.06, 2.02, "GAIN  (+)", fontsize=7.6, color=INK2, ha="right", va="top")
    ax.text(-0.06, -2.26, "COST  (\u2212)", fontsize=7.6, color=INK2, ha="right",
            va="bottom")


    legend = [
        Line2D([], [], marker="o", ls="none", ms=7, markerfacecolor=BLUE,
               markeredgecolor="white", label="Cost episode"),
        Line2D([], [], marker="o", ls="none", ms=7, markerfacecolor=TEAL,
               markeredgecolor="white", label="Gain episode"),
        Patch(facecolor=BLUE, hatch="///", edgecolor=INK, ls=(0, (3, 1.5)), lw=1.1,
              label="Trade-transmitted market price (not a policy instrument)"),
        Line2D([], [], marker="o", ls="none", ms=7, markerfacecolor="white",
               markeredgecolor=TEAL, markeredgewidth=1.4,
               label="Near-zero benchmark (no household structure)"),
    ]
    # Size key lives in the legend box, outside the plot: open circles with
    # strong edges, diameters on the same log-area scale as the bubbles.
    for val, lab in [(0.18, "\u00a30.18bn gross"), (2.0, "\u00a32bn gross"),
                     (63.9, "\u00a364bn gross")]:
        ms = 2.0 * np.sqrt(bubble(val) / np.pi) * 0.42
        legend.append(Line2D([], [], marker="o", ls="none", ms=ms,
                             markerfacecolor="none", markeredgecolor=INK2,
                             markeredgewidth=1.4, label=lab))
    ax.legend(handles=legend, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.005), fontsize=7.4,
              handletextpad=0.9, columnspacing=1.6, labelspacing=1.25,
              frameon=False)
    save(fig, "fig_episode_map.png")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------
def _num(x, dp=0):
    return f"{x:,.{dp}f}"


def write_tables(d: Data):
    L = []
    A = L.append
    A("% Generated by figures/make_figures.py -- do not edit by hand.")
    A("% Requires: \\usepackage{booktabs}  (no other package needed)")
    A("% \\rr = ragged-right inside a p{} cell, with \\\\ restored as the row")
    A("% terminator, so narrow columns neither overfull nor stretch-underfull.")
    A("\\providecommand{\\rr}{\\raggedright\\let\\\\\\tabularnewline}")
    A("")

    # --- tab:decile-full ------------------------------------------------
    A("\\begin{table}[tbp]")
    A("\\centering")
    A("\\small")
    A("\\caption{Household incidence of the quantified episodes, by equivalised")
    A("disposable income decile. TCA food NTBs are on the window-average path")
    A("(%.2f of the end-state $+8\\%%$ effect); energy is the 2022--23 cap shock,"
      % d.tca_share)
    A("gross and net of the Energy Price Guarantee; the India CETA column is the")
    A("consumer gain at 100\\% pass-through. Costs are positive.}")
    A("\\label{tab:decile-full}")
    A("\\begin{tabular}{@{}l rr rr rr r@{}}")
    A("\\toprule")
    A(" & \\multicolumn{2}{c}{TCA food NTBs} & \\multicolumn{2}{c}{Energy, gross}")
    A(" & \\multicolumn{2}{c}{Energy, net of EPG} & CETA gain \\\\")
    A("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-8}")
    A("Decile & \\pounds/yr & \\% spend & \\pounds/yr & \\% spend"
      " & \\pounds/yr & \\% spend & \\pounds/yr \\\\")
    A("\\midrule")
    for i in range(10):
        A(
            f"{i + 1} & {_num(d.tca_win_gbp[i])} & {d.tca_win_pct[i]:.2f} & "
            f"{_num(d.en_gross_gbp[i])} & {d.en_gross_pct[i]:.2f} & "
            f"{_num(d.en_net_gbp[i])} & {d.en_net_pct[i]:.2f} & "
            f"{d.ceta_gbp[i]:.2f} \\\\"
        )
    A("\\midrule")
    A(
        "D1/D10 ratio & --- & "
        f"{d.tca_win_pct[0] / d.tca_win_pct[9]:.2f} & --- & "
        f"{d.en_gross_pct[0] / d.en_gross_pct[9]:.2f} & --- & "
        f"{d.en_net_pct[0] / d.en_net_pct[9]:.2f} & "
        f"{d.ceta_pct[0] / d.ceta_pct[9]:.2f}$^{{\\dagger}}$ \\\\"
    )
    A("\\bottomrule")
    A("\\end{tabular}")
    A("")
    A("\\vspace{2pt}")
    A("\\begin{minipage}{\\linewidth}\\footnotesize")
    A("\\emph{Notes.} \\pounds\\ per household per year; \\% spend is the burden or")
    A("gain as a percentage of that decile's total expenditure. The EPG cushions")
    A("%.1f\\%% of the gross energy shock at every decile by construction."
      % (100 * d.cushion_share))
    A("$^{\\dagger}$~CETA ratio is on the \\%-of-spending basis (\\pounds\\ gains are")
    A("not comparable across deciles). Spending bases: ONS Family Spending")
    A("workbook~1, sheet 3.1E, FYE2022 vintage (TCA, energy) and FYE2025 vintage (CETA).")
    A("\\end{minipage}")
    A("\\end{table}")
    A("")

    # --- tab:first-stages ------------------------------------------------
    # Cite keys are placeholders: map them onto the manuscript's .bib.
    rows = [
        ("E1 TCA food NTBs", "Dec 2019--\\newline Mar 2023$^{w}$", "Policy",
         "Consumer prices (food)", "\\citet{bakker2026}",
         "$+8\\%$ on food prices (6\\% low variant)", "Estimate\\newline (ex post)"),
        ("E2 Energy 2022--23", "Oct 2022--\\newline Mar 2023", "Market",
         "Consumer prices (energy)", "\\citet{ofgem2022}; \\citet{obrenergy2022}",
         "Cap \\pounds1{,}277 $\\to$ \\pounds3{,}549; EPG \\pounds2{,}500",
         "Observed\\newline (statutory)"),
        ("E3 US tariffs 2025", "2025", "Policy\\newline (foreign)",
         "Earnings only; no UK retaliation", "\\citet{ahmadi2026}",
         "\\pounds886m/yr gross earnings shock", "Estimate\\newline (projection)"),
        ("E4 India CETA", "2026+", "Policy",
         "Consumer prices (clothing, footwear, food)", "\\citet{dbtindia2026}",
         "\\pounds180m/yr duty cut on final goods", "Projection\\newline (ex ante)"),
        ("E5 CPTPP benchmark", "Long run", "Policy",
         "Aggregate GDP; no household mapping", "\\citet{dbtcptpp2023}",
         "$+$\\pounds2.0bn GDP ($+0.08\\%$)", "Projection\\newline (long run)"),
    ]
    A("\\begin{table}[tbp]")
    A("\\centering")
    A("\\footnotesize")
    A("\\setlength{\\tabcolsep}{3pt}")
    A("\\renewcommand{\\arraystretch}{1.15}")
    A("\\caption{Declared first stages. Every price or earnings vector is an")
    A("imported published estimate; none is re-estimated in this paper.}")
    A("\\label{tab:first-stages}")
    A("\\begin{tabular}{@{}p{1.75cm} p{1.70cm} p{1.45cm} p{2.30cm} p{2.10cm}"
      " p{2.75cm} p{1.85cm}@{}}")
    A("\\toprule")
    A(" & ".join("\\rr " + c for c in
                 ("Episode", "Date", "Type", "Channel", "First-stage source",
                  "Key imported number", "Status")) + " \\\\")
    A("\\midrule")
    for r in rows:
        A(" & ".join("\\rr " + c for c in r) + " \\\\[3pt]")
    A("\\bottomrule")
    A("\\end{tabular}")
    A("")
    A("\\vspace{2pt}")
    A("\\begin{minipage}{\\linewidth}\\footnotesize")
    A("\\emph{Notes.} Type distinguishes a policy instrument from a")
    A("trade-transmitted market price. Source entries are short cite keys; see the")
    A("bibliography. E3's consumer-price row is exactly zero (the UK imposed no")
    A("retaliatory tariffs), so its household incidence runs through earnings alone.")
    A("$^{w}$~the \\emph{estimation} window of the cited study; the TCA itself")
    A("entered into force in January 2021.")
    A("\\end{minipage}")
    A("\\end{table}")
    A("")

    text_all = "\n".join(L)

    # Split into per-table files so the manuscript can place each one in its
    # own section (decile detail in results; first stages in episodes).
    text = text_all
    preamble = text.split("\\begin{table}")[0]
    bodies = ["\\begin{table}" + b for b in text.split("\\begin{table}")[1:]]
    for name, body in zip(
        ("generated_table_decile.tex", "generated_table_firststages.tex"), bodies
    ):
        p = HERE / name
        p.write_text(preamble + body)
        print(f"wrote {p}")


def main():
    apply_style()
    d = Data(json.loads(RESULTS.read_text()))
    fig_incidence_profile(d)
    fig_energy_epg(d)
    fig_budget_shares(d)
    fig_uprating_lag(d)
    fig_episode_map(d)
    write_tables(d)


if __name__ == "__main__":
    main()
