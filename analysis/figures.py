"""Paper figures from results/*.json (PolicyEngine house style via figstyle).

Stubs where results are pending; run analysis/run_scenarios.py first.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("results")
FIGURES = Path("results/figures")


def load(name: str) -> dict:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run analysis/run_scenarios.py first")
    return json.loads(path.read_text())


def decile_mc(result: dict) -> tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo mean and SD of the decile income-change profile."""
    draws = np.array(
        [[d["decile_income_change"][str(k)] for k in range(1, 11)] for d in result["draws"]]
    )
    sd = draws.std(axis=0, ddof=1) if len(draws) > 1 else np.zeros(10)
    return draws.mean(axis=0), sd


def fig_decile_by_margin(tariff: str = "full_tariff") -> None:
    """Decile income change under the three adjustment margins (MC means,
    shaded +/- 1 SD across draws for the stochastic margins)."""
    figstyle.apply_style()
    fig, ax = plt.subplots(figsize=figstyle.SINGLE)
    x = np.arange(1, 11)
    for margin, colour in zip(
        ("displacement", "wage_cut", "inactivity"), figstyle.SERIES
    ):
        mean, sd = decile_mc(load(f"{tariff}_{margin}"))
        ax.plot(x, mean, marker="o", color=colour, label=margin.replace("_", " "))
        if sd.any():
            ax.fill_between(x, mean - sd, mean + sd, color=colour, alpha=0.15, lw=0)
    figstyle.decile_ax(ax, "Mean change in household disposable income (£/year)")
    figstyle.legend_below(ax, ncol=3)
    FIGURES.mkdir(parents=True, exist_ok=True)
    figstyle.save(fig, FIGURES / f"decile_by_margin_{tariff}.png")


def fig_exchequer_draws() -> None:
    """Draw-distribution of displacement costs with wage-cut mean and SD."""
    figstyle.apply_style()
    fig, ax = plt.subplots(figsize=figstyle.SINGLE)
    labels, data = [], []
    for tariff, label in (("full_tariff", "Full tariffs"), ("epd", "EPD")):
        result = load(f"{tariff}_displacement")
        data.append([d["exchequer_cost"] / 1e9 for d in result["draws"]])
        labels.append(label)
    bp = ax.boxplot(
        data, tick_labels=labels, patch_artist=True, widths=0.45,
        medianprops={"color": figstyle.BLUE_PRESSED},
        boxprops={"facecolor": figstyle.BLUE_LIGHT, "edgecolor": figstyle.BLUE},
        whiskerprops={"color": figstyle.BLUE}, capprops={"color": figstyle.BLUE},
        flierprops={"markeredgecolor": figstyle.GRAY, "markersize": 4},
    )
    rng = np.random.default_rng(0)
    for i, draws in enumerate(data, start=1):
        ax.scatter(
            i + rng.uniform(-0.08, 0.08, len(draws)), draws,
            s=10, color=figstyle.BLUE, alpha=0.45, zorder=3,
        )
    for i, tariff in enumerate(("full_tariff", "epd"), start=1):
        wc_result = load(f"{tariff}_wage_cut")
        wc = wc_result["exchequer_cost_mean"] / 1e9
        wc_sd = wc_result["exchequer_cost_sd"] / 1e9
        ax.errorbar(
            i, wc, yerr=wc_sd, fmt="D", color=figstyle.TEAL_PRESSED,
            capsize=4, zorder=4,
            label="wage-cut margin (mean ± assignment SD)" if i == 1 else None,
        )
    ax.set_ylabel("Exchequer cost (£bn/year)")
    ax.set_xlabel("Displacement-margin Monte Carlo draws")
    figstyle.legend_below(ax, ncol=1)
    FIGURES.mkdir(parents=True, exist_ok=True)
    figstyle.save(fig, FIGURES / "exchequer_cost_draws.png")


def fig_epd_counterfactual() -> None:
    """Exchequer cost, full tariff vs EPD, by margin (mean +/- SD bars)."""
    figstyle.apply_style()
    fig, ax = plt.subplots(figsize=figstyle.SINGLE)
    margins = ("displacement", "wage_cut", "inactivity")
    for i, (tariff, colour) in enumerate(
        (("full_tariff", figstyle.BLUE), ("epd", figstyle.TEAL))
    ):
        means = [load(f"{tariff}_{m}")["exchequer_cost_mean"] / 1e9 for m in margins]
        sds = [load(f"{tariff}_{m}")["exchequer_cost_sd"] / 1e9 for m in margins]
        x = [j + (i - 0.5) * 0.35 for j in range(len(margins))]
        ax.bar(x, means, width=0.35, yerr=sds, color=colour,
               label="Full tariffs" if tariff == "full_tariff" else "EPD")
    ax.set_xticks(range(len(margins)))
    ax.set_xticklabels([m.replace("_", " ") for m in margins])
    ax.set_ylabel("Exchequer cost (£bn/year)")
    figstyle.legend_below(ax, ncol=2)
    FIGURES.mkdir(parents=True, exist_ok=True)
    figstyle.save(fig, FIGURES / "epd_counterfactual.png")


if __name__ == "__main__":
    fig_decile_by_margin()
    fig_epd_counterfactual()
    fig_exchequer_draws()
