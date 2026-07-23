"""Write the reproducible parameter design used for structured sensitivity.

This is data-free and intentionally does not run PolicyEngine. It creates the
design that an expensive licensed-data run can consume later.
"""

from pathlib import Path

from uk_trade_shock_study.uncertainty import latin_hypercube


def main() -> None:
    design = latin_hypercube(
        {
            "elasticity": (0.4, 3.0),
            "wage_bill_incidence": (0.5, 1.0),
            "uc_takeup": (0.55, 0.95),
            "reallocation_penalty": (0.14, 0.283),
            "displacement_share": (0.0, 1.0),
        },
        n_draws=500,
        seed=20260723,
    )
    out = Path("results/parameter_uncertainty_design.csv")
    out.parent.mkdir(exist_ok=True)
    design.to_csv(out, index=False)
    print(f"wrote {out} ({len(design)} draws)")


if __name__ == "__main__":
    main()
