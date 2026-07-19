"""Leontief upstream (supply-chain) extension of the direct tariff shock.

APPENDIX SCENARIO. The core design covers the direct exposed-sector channel
only (exposure.py). This module bounds the first omitted channel of
discussion limitation (4): upstream input-supply propagation through the
domestic production network (Barrot & Sauvagnat 2016; Acemoglu, Akcigit &
Kerr 2016).

Method
------
Let f be the vector of direct export-demand falls in GROSS-OUTPUT terms
(f_j = elasticity x tariff_j x US exports of division j, the same primitives
as Equation (1) of the paper, in pounds rather than as a wage-bill share).
With A the DOMESTIC technical-coefficient matrix from the ONS UK
input-output analytical tables (product-by-product, 2022 revised edition,
sheet "IOT": the domestic use matrix at basic prices, £m) and
L = (I - A)^{-1} the Leontief inverse, the upstream requirement falls are

    delta_g = (L - I) f          (direct round excluded: f itself is the
                                  direct channel already in the paper).

Each supplying product k converts its output fall to an earnings loss via
its compensation-of-employees/gross-output ratio (D1 and P1 rows of the same
IOT sheet), i.e. earnings loss_k = delta_g_k x (CoE_k / GO_k), and to an
earnings-SHOCK RATE (fraction of the product wage bill, comparable to s_j)
of s_k^up = passthrough x delta_g_k / GO_k. The person-level scenario then
runs on s^total = s^direct + s^upstream.

CPA -> SIC 2007 mapping
-----------------------
The 104 IOAT products are CPA 2008 groups, which align with SIC 2007 at the
2-digit division except where ONS aggregates or splits:
- splits WITHIN a division (e.g. CPA_C101..C109 in division 10, CPA_C20A/B/C
  in 20, CPA_C241_3/C244_5 in 24, CPA_C301/C303/C30OTHER in 30): the
  division's direct fall f_j is spread over its products in proportion to
  product gross output, and the division's upstream rate is the
  gross-output-weighted aggregate sum(delta_g_k)/sum(GO_k) over its products;
- groups SPANNING divisions (CPA_B06 & B07, CPA_C11.01-6 & C12,
  CPA_F41, F42 & F43, CPA_J59 & J60, CPA_Q87 & Q88): every member division
  inherits the group's rate (uniform-rate assumption within the group).
The mapping is parsed from the CPA codes themselves (section letter +
leading two digits of each '&'/','-separated token) — see
_divisions_of_product; a hand-checked override table catches the
non-parseable codes.

The ONS xlsx is cached at data/iot2022revisedproduct.xlsx (source: ONS,
"UK input-output analytical tables: product by product", 2022 revised,
released Feb 2025; downloaded via
https://www.ons.gov.uk/file?uri=/economy/nationalaccounts/supplyandusetables/
datasets/ukinputoutputanalyticaltablesdetailed/2022revised/
iot2022revisedproduct.xlsx).

US export values by division reuse analysis/build_trade_by_sic.py (HMRC
uktradeinfo cached pulls) — the identical numerator as the intensity build.
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from uk_trade_shock_study.exposure import (
    DEFAULT_ELASTICITY,
    DEFAULT_PASSTHROUGH,
    sector_earnings_shocks,
    tariff_rates,
)
from uk_trade_shock_study.shocks import DEFAULT_UC_TAKEUP

ROOT = Path(__file__).resolve().parent.parent
IOT_XLSX = ROOT / "data" / "iot2022revisedproduct.xlsx"
IOT_SHEET = "IOT"

#: CPA codes whose SIC divisions cannot be parsed positionally.
_PRODUCT_DIVISION_OVERRIDES = {
    "CPA_C11.01-6 & C12": (11, 12),
    "CPA_C23OTHER": (23,),
    "CPA_C30OTHER": (30,),
    "CPA_C33OTHER": (33,),
    "CPA_F41, F42 & F43": (41, 42, 43),
    "CPA_K65.1-2 & K65.3": (65,),
    "CPA_L683": (68,),
    "CPA_L68A": (68,),
    "CPA_L68BXL683": (68,),
}


def _divisions_of_product(code: str) -> tuple[int, ...]:
    """SIC 2007 division(s) of a CPA product code (e.g. 'CPA_C102_3' -> (10,))."""
    if code in _PRODUCT_DIVISION_OVERRIDES:
        return _PRODUCT_DIVISION_OVERRIDES[code]
    divisions = []
    for token in re.split(r"[&,]", code.removeprefix("CPA_")):
        m = re.match(r"\s*[A-Z](\d{2})", token.strip())
        if m:
            divisions.append(int(m.group(1)))
    if not divisions:
        raise ValueError(f"cannot parse SIC divisions from CPA code {code!r}")
    return tuple(dict.fromkeys(divisions))


@dataclass(frozen=True)
class IOTables:
    """Domestic product-by-product IO system (£m, basic prices)."""

    products: tuple[str, ...]
    intermediate: np.ndarray  # Z, products x products, domestic use
    gross_output: np.ndarray  # P1 row
    compensation: np.ndarray  # D1 row (compensation of employees)

    @property
    def technical_coefficients(self) -> np.ndarray:
        """A = Z g^{-1} (columns divided by product gross output)."""
        g = np.where(self.gross_output > 0, self.gross_output, np.inf)
        return self.intermediate / g[np.newaxis, :]

    @property
    def leontief_inverse(self) -> np.ndarray:
        return np.linalg.inv(np.eye(len(self.products)) - self.technical_coefficients)

    @property
    def coe_ratio(self) -> np.ndarray:
        """Compensation-of-employees share of gross output per product."""
        g = np.where(self.gross_output > 0, self.gross_output, np.inf)
        return self.compensation / g


def load_iot(path: str | Path = IOT_XLSX) -> IOTables:
    """Parse the ONS IOAT product-by-product workbook (sheet 'IOT')."""
    import openpyxl

    ws = openpyxl.load_workbook(str(path), read_only=True)[IOT_SHEET]
    rows = list(ws.iter_rows(values_only=True))
    # The amplification factor is only meaningful on the DOMESTIC use matrix:
    # the total (incl. imports) matrix would attribute imported intermediates
    # to UK suppliers and inflate upstream earnings losses. The ONS workbook
    # carries the total-vs-domestic distinction in the sheet title (and keeps
    # the import content on separate 'Imports use ...' sheets).
    title = " ".join(str(c) for r in rows[:3] for c in r if c).lower()
    if "domestic use" not in title:
        raise ValueError(
            f"IOT sheet {IOT_SHEET!r} is not labelled a DOMESTIC use matrix "
            f"(title: {title!r}); the upstream amplification would be "
            "overstated by imported intermediate content."
        )
    header = [c for c in rows[3][2:] if c]
    body = {r[0]: r for r in rows[6:] if r[0]}
    products = tuple(k for k in body if str(k).startswith("CPA_"))
    n = len(products)
    if list(header[:n]) != list(products):
        raise ValueError("IOT sheet rows and columns are not identically ordered")
    Z = np.array([[float(body[p][2 + j]) for j in range(n)] for p in products])
    go = np.array([float(body["P1"][2 + j]) for j in range(n)])
    coe = np.array([float(body["D1"][2 + j]) for j in range(n)])
    # Cross-check: domestic intermediate use of each product cannot exceed
    # that product's domestic gross output (it would under a total-use matrix,
    # which includes imported supply of the same product).
    row_use = Z.sum(axis=1)
    if np.any(row_use > go * (1.0 + 1e-6)):
        bad = [products[i] for i in np.flatnonzero(row_use > go * (1.0 + 1e-6))]
        raise ValueError(
            "parsed IOT matrix has intermediate use exceeding domestic gross "
            f"output for {bad}: this indicates a TOTAL (incl. imports) use "
            "matrix, which would inflate the upstream amplification factor."
        )
    return IOTables(products=products, intermediate=Z, gross_output=go, compensation=coe)


def upstream_output_falls(iot: IOTables, final_demand_fall: np.ndarray) -> np.ndarray:
    """delta_g = (L - I) f: upstream gross-output requirement falls (£m).

    The direct round f is excluded — it is the paper's direct channel.
    """
    L = iot.leontief_inverse
    return (L - np.eye(len(iot.products))) @ np.asarray(final_demand_fall, float)


def load_us_exports_by_division() -> dict[int, float]:
    """US goods exports 2024 (£) by SIC division, via the cached HMRC pulls.

    Reuses analysis/build_trade_by_sic.py (same numerator as the packaged
    intensity table) by loading it as a module from its file path — the
    analysis directory is not a package.
    """
    spec = importlib.util.spec_from_file_location(
        "build_trade_by_sic", ROOT / "analysis" / "build_trade_by_sic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fetch_us_exports_by_division()


def direct_final_demand_falls(
    iot: IOTables,
    tariff_scenario: str,
    elasticity: float = DEFAULT_ELASTICITY,
    exports_by_division: dict[int, float] | None = None,
) -> np.ndarray:
    """f: direct export-demand falls in gross-output terms, per IO product (£m).

    f_j = elasticity x tariff_j x US exports_j (the paper's primitives in
    pounds); a division's fall is spread over its IO products in proportion
    to product gross output.
    """
    if exports_by_division is None:
        exports_by_division = load_us_exports_by_division()
    rates = tariff_rates(tariff_scenario)
    f = np.zeros(len(iot.products))
    prod_divs = [_divisions_of_product(p) for p in iot.products]
    for division, exports in exports_by_division.items():
        if division not in rates.index:
            continue
        fall = elasticity * float(rates.loc[division]) * exports / 1e6  # £ -> £m
        members = [i for i, divs in enumerate(prod_divs) if division in divs]
        if not members:
            continue
        go = iot.gross_output[members]
        f[members] += fall * (go / go.sum() if go.sum() > 0 else 1.0 / len(members))
    return f


def upstream_sector_shocks(
    tariff_scenario: str,
    elasticity: float = DEFAULT_ELASTICITY,
    passthrough: float = DEFAULT_PASSTHROUGH,
    iot: IOTables | None = None,
    exports_by_division: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Upstream earnings-shock rate and earnings loss per SIC division.

    Returns a DataFrame indexed by sic_division with columns:
    upstream_output_fall (£m), upstream_earnings_loss (£m),
    upstream_shock (rate: passthrough x sum delta_g / sum GO over the
    division's products).
    """
    if iot is None:
        iot = load_iot()
    f = direct_final_demand_falls(iot, tariff_scenario, elasticity, exports_by_division)
    delta_g = upstream_output_falls(iot, f)
    earnings = delta_g * iot.coe_ratio

    # Two distinct accumulations.
    #
    # (a) RATES. A CPA group spanning several SIC divisions imposes its rate
    #     on every member division (the uniform-rate assumption documented in
    #     the module docstring), so the rate numerator/denominator book the
    #     FULL delta_g/GO to each member. Unchanged.
    #
    # (b) LEVELS (upstream_output_fall, upstream_earnings_loss). These are
    #     additive quantities and must SUM to the economy aggregate, so a
    #     spanning group's level is SPLIT across its member divisions rather
    #     than replicated. The split key is each member division's share of
    #     the gross output of the products it holds exclusively (products
    #     mapping to that division alone); if none of the member divisions
    #     has an exclusive product, the group is split equally.
    prod_divs = [_divisions_of_product(p) for p in iot.products]
    solo_go: dict[int, float] = {}
    for i, divs in enumerate(prod_divs):
        if len(divs) == 1:
            solo_go[divs[0]] = solo_go.get(divs[0], 0.0) + float(iot.gross_output[i])

    records: dict[int, dict[str, float]] = {}
    for i, divs in enumerate(prod_divs):
        keys = np.array([solo_go.get(d, 0.0) for d in divs], dtype=float)
        weights = (
            keys / keys.sum() if keys.sum() > 0 else np.full(len(divs), 1.0 / len(divs))
        )
        for division, w in zip(divs, weights):
            rec = records.setdefault(
                division,
                {
                    "upstream_output_fall": 0.0,
                    "upstream_earnings_loss": 0.0,
                    "rate_output_fall": 0.0,
                    "go": 0.0,
                },
            )
            rec["upstream_output_fall"] += float(delta_g[i]) * float(w)
            rec["upstream_earnings_loss"] += float(earnings[i]) * float(w)
            rec["rate_output_fall"] += float(delta_g[i])
            rec["go"] += float(iot.gross_output[i])
    table = pd.DataFrame.from_dict(records, orient="index").rename_axis("sic_division")
    table["upstream_shock"] = passthrough * table["rate_output_fall"] / table["go"]
    return table.drop(columns=["go", "rate_output_fall"]).sort_index()


def total_sector_shocks(
    tariff_scenario: str,
    elasticity: float = DEFAULT_ELASTICITY,
    passthrough: float = DEFAULT_PASSTHROUGH,
    iot: IOTables | None = None,
    exports_by_division: dict[int, float] | None = None,
) -> pd.DataFrame:
    """s^total = s^direct + s^upstream per SIC division.

    Direct shocks are exposure.sector_earnings_shocks (wage-bill-share
    rates); upstream rates come from the Leontief round. Divisions with only
    an upstream shock (services) enter with s_direct = 0.
    """
    upstream = upstream_sector_shocks(
        tariff_scenario, elasticity, passthrough, iot, exports_by_division
    )
    direct = sector_earnings_shocks(tariff_scenario, elasticity, passthrough)
    index = upstream.index.union(direct.index)
    table = pd.DataFrame(index=index).rename_axis("sic_division")
    table["direct_shock"] = direct.reindex(index).fillna(0.0)
    table["upstream_shock"] = upstream["upstream_shock"].reindex(index).fillna(0.0)
    table["total_shock"] = (table["direct_shock"] + table["upstream_shock"]).clip(0.0, 1.0)
    table["upstream_earnings_loss"] = (
        upstream["upstream_earnings_loss"].reindex(index).fillna(0.0)
    )
    return table


def amplification_summary(
    tariff_scenario: str,
    elasticity: float = DEFAULT_ELASTICITY,
    passthrough: float = DEFAULT_PASSTHROUGH,
    iot: IOTables | None = None,
    exports_by_division: dict[int, float] | None = None,
) -> dict:
    """IO-accounts amplification: total/direct earnings loss, top suppliers.

    Both channels are valued with the SAME IO earnings conversion
    (delta output x CoE/GO), so the ratio is a pure network statistic,
    independent of the FRS realisation.
    """
    if iot is None:
        iot = load_iot()
    f = direct_final_demand_falls(iot, tariff_scenario, elasticity, exports_by_division)
    delta_g = upstream_output_falls(iot, f)
    direct_earnings = float((f * iot.coe_ratio).sum()) * passthrough
    upstream_earnings = float((delta_g * iot.coe_ratio).sum()) * passthrough
    upstream = upstream_sector_shocks(
        tariff_scenario, elasticity, passthrough, iot, exports_by_division
    )
    top = upstream.sort_values("upstream_earnings_loss", ascending=False).head(12)
    return {
        "tariff_scenario": tariff_scenario,
        "direct_earnings_loss_gbp_m": direct_earnings,
        "upstream_earnings_loss_gbp_m": upstream_earnings,
        "amplification_factor": (direct_earnings + upstream_earnings) / direct_earnings,
        "top_upstream_divisions": {
            int(d): {
                "upstream_earnings_loss_gbp_m": float(r["upstream_earnings_loss"]),
                "upstream_shock": float(r["upstream_shock"]),
            }
            for d, r in top.iterrows()
        },
    }


def person_total_shock(
    sic_division: np.ndarray | pd.Series,
    tariff_scenario: str,
    elasticity: float = DEFAULT_ELASTICITY,
    passthrough: float = DEFAULT_PASSTHROUGH,
    shocks: pd.DataFrame | None = None,
) -> np.ndarray:
    """Per-person s^total from SIC division codes (NaN/unmatched -> 0)."""
    if shocks is None:
        shocks = total_sector_shocks(tariff_scenario, elasticity, passthrough)
    codes = pd.to_numeric(pd.Series(np.asarray(sic_division)), errors="coerce")
    return codes.map(shocks["total_shock"]).fillna(0.0).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Margin realisation with an EXPLICIT per-person shock array. These mirror
# shocks.draw_displaced / apply_wage_cut but take the shock vector directly,
# so the supply-chain scenario can run s^total without altering the core
# scenario machinery (shocks.py derives its shock internally from the
# tariff scenario, which covers the direct channel only).
# ---------------------------------------------------------------------------


def draw_displaced_with_shock(
    persons: pd.DataFrame, shock: np.ndarray, seed: int = 0
) -> np.ndarray:
    """Displacement mask for a per-person shock vector (uniform ordering keys).

    Same contract as shocks.draw_displaced: per-division weighted quota
    shock_j x weighted employee count, weights consumed in uniform random
    order, quota-crossing record included with the fractional probability.
    """
    rng = np.random.default_rng(seed)
    employed = persons["employment_income"].to_numpy(dtype=float) > 0
    weight = persons["weight"].to_numpy(dtype=float)
    division = pd.to_numeric(persons["sic_division"], errors="coerce").to_numpy(dtype=float)
    shock = np.asarray(shock, dtype=float)

    displaced = np.zeros(len(persons), dtype=bool)
    for d in np.unique(division[employed & (shock > 0)]):
        members = np.flatnonzero(employed & (division == d))
        quota = float(shock[members[0]]) * float(weight[members].sum())
        if quota <= 0:
            continue
        chosen = rng.permutation(members)
        cum = np.cumsum(weight[chosen])
        displaced[chosen[cum <= quota]] = True
        crossing = np.searchsorted(cum, quota)
        if crossing < len(chosen) and cum[crossing] > quota:
            shortfall = quota - (cum[crossing - 1] if crossing else 0.0)
            if rng.random() < shortfall / weight[chosen[crossing]]:
                displaced[chosen[crossing]] = True
    return displaced


def apply_displacement_with_shock(
    persons: pd.DataFrame,
    shock: np.ndarray,
    seed: int = 0,
    uc_takeup: float = DEFAULT_UC_TAKEUP,
) -> pd.DataFrame:
    """Displacement margin on an explicit shock vector (no inactivity flow)."""
    shocked = persons.copy()
    displaced = draw_displaced_with_shock(persons, shock, seed=seed)
    shocked["displaced"] = displaced
    shocked["inactive"] = np.zeros(len(persons), dtype=bool)
    earnings = shocked["employment_income"].to_numpy(dtype=float)
    shocked["employment_income"] = np.where(displaced, 0.0, earnings)
    # carried to build_shocked_simulation for the post-shock UC take-up
    # re-draw (see shocks.DEFAULT_UC_TAKEUP); seed must vary across draws so
    # the take-up stream is not frozen at seed 0.
    shocked.attrs["uc_takeup"] = float(uc_takeup)
    shocked.attrs["seed"] = int(seed)
    return shocked


def apply_wage_cut_with_shock(persons: pd.DataFrame, shock: np.ndarray) -> pd.DataFrame:
    """Wage-cut margin on an explicit shock vector, conservation asserted."""
    shocked = persons.copy()
    earnings = shocked["employment_income"].to_numpy(dtype=float)
    weight = shocked["weight"].to_numpy(dtype=float)
    employed = earnings > 0
    shock = np.asarray(shock, dtype=float)
    if (shock[employed] >= 1.0).any():
        raise ValueError("total sector shock >= 100% of earnings for some workers")
    target = float((shock * earnings * weight)[employed].sum())
    new = np.where(employed, earnings * (1.0 - shock), earnings)
    realised = float(((earnings - new) * weight)[employed].sum())
    if not np.isclose(realised, target, rtol=1e-9):
        raise AssertionError("wage-bill conservation failed")
    shocked["employment_income"] = new
    shocked["displaced"] = np.zeros(len(persons), dtype=bool)
    shocked["inactive"] = np.zeros(len(persons), dtype=bool)
    # no new claimants on the wage-cut margin: nothing is re-drawn
    shocked.attrs["uc_takeup"] = float(DEFAULT_UC_TAKEUP)
    shocked.attrs["seed"] = 0
    return shocked
