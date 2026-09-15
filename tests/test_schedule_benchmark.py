"""The statutory schedule benchmark, and the periodicity guards around it.

``schedule_benchmark_block`` feeds \\ScheduleMarginalRate,
\\ScheduleAverageRate and \\ScheduleImpliedGap, the one-worker arithmetic the
manuscript's mechanism argument is checked against. It used to:

- alias the income tax personal allowance as the National Insurance primary
  threshold (they coincide only by arithmetic coincidence in some years);
- ignore the upper earnings limit and the higher-rate band entirely, so it was
  correct only between the allowance and the higher-rate threshold;
- store a ``higher_rate_threshold`` in its parameters dict that nothing read.

None of the three raised. Every test below fails under that implementation.
"""

import json
from pathlib import Path

import pytest

from analysis.referee_fixes import (
    EXPOSED_MEAN_EARNINGS,
    NI_PRIMARY_THRESHOLD_WEEKLY_RANGE,
    NI_UPPER_EARNINGS_LIMIT_WEEKLY_RANGE,
    PARAMETER_SOURCES,
    PERSONAL_ALLOWANCE_TAPER_THRESHOLD,
    SCHEDULE_REQUIRED_PARAMETERS,
    UC_STANDARD_ALLOWANCE_MONTHLY_RANGE,
    MONTHLY_UC_NOTES,
    _check_periodic_amount,
    employee_ni,
    income_tax,
    monthly_uc_block,
    schedule_benchmark_block,
)

# 2025-26 statutory schedule, with the personal allowance and the NI primary
# threshold DELIBERATELY separated where a test needs them to diverge.
BASE = {
    "representative_earnings": EXPOSED_MEAN_EARNINGS,
    "personal_allowance": 12_570.0,
    "basic_rate": 0.20,
    "higher_rate": 0.40,
    "basic_rate_limit": 37_700.0,
    "higher_rate_threshold": 50_270.0,
    "ni_employee_main": 0.08,
    "ni_employee_above_uel": 0.02,
    "ni_primary_threshold_annual": 12_570.0,
    "ni_upper_earnings_limit_annual": 50_270.0,
    "standard_allowance_single_25_plus_month": 424.90,
}


def block(**overrides) -> dict:
    return schedule_benchmark_block({"parameters": {**BASE, **overrides}})


# ---------------------------------------------------------------------------
# The primary threshold is its own parameter, not an alias of the allowance
# ---------------------------------------------------------------------------


def test_ni_uses_the_primary_threshold_not_the_personal_allowance() -> None:
    """A year in which the two diverge. The old code emitted wrong rates here.

    Personal allowance 12,570 but a primary threshold of 9,000: employee NI is
    charged on 21,000 of earnings, not on the 17,430 of TAXABLE income the old
    implementation used.
    """
    out = block(
        representative_earnings=30_000.0, ni_primary_threshold_annual=9_000.0
    )
    assert out["income_tax"] == pytest.approx(0.20 * (30_000.0 - 12_570.0))
    assert out["employee_national_insurance"] == pytest.approx(
        0.08 * (30_000.0 - 9_000.0)
    )
    # what the allowance-aliasing implementation would have produced
    aliased = 0.08 * (30_000.0 - 12_570.0)
    assert out["employee_national_insurance"] != pytest.approx(aliased)
    assert out["average_deduction_rate"] == pytest.approx(
        (0.20 * 17_430.0 + 0.08 * 21_000.0) / 30_000.0
    )
    assert out["parameters"]["ni_primary_threshold_annual"] == 9_000.0


def test_a_primary_threshold_above_earnings_charges_no_ni() -> None:
    out = block(representative_earnings=20_000.0, ni_primary_threshold_annual=25_000.0)
    assert out["employee_national_insurance"] == 0.0
    assert out["marginal_deduction_rate"] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Higher-rate band and upper earnings limit
# ---------------------------------------------------------------------------


def test_earnings_above_the_higher_rate_threshold_and_above_the_uel() -> None:
    """Both bands the old implementation dropped, in one worker.

    At 80,000: 37,700 of taxable income at 20 per cent and 29,730 at 40; NI at
    8 per cent on 37,700 and at 2 per cent on the 29,730 above the UEL. The old
    code taxed the whole taxable slice at the basic rate and charged NI at the
    main rate on all of it, with no error.
    """
    out = block(representative_earnings=80_000.0)
    assert out["income_tax"] == pytest.approx(0.20 * 37_700.0 + 0.40 * 29_730.0)
    assert out["employee_national_insurance"] == pytest.approx(
        0.08 * 37_700.0 + 0.02 * 29_730.0
    )
    assert out["marginal_deduction_rate"] == pytest.approx(0.40 + 0.02)
    assert out["average_deduction_rate"] == pytest.approx(
        (19_432.0 + 3_610.60) / 80_000.0
    )
    assert out["implied_gap_percentage_points"] == pytest.approx(
        100.0 * (0.42 - (19_432.0 + 3_610.60) / 80_000.0)
    )
    assert out["income_tax_band"] == "higher-rate"
    assert "upper earnings limit" in out["national_insurance_band"]
    # the old implementation's answers, explicitly excluded
    assert out["income_tax"] != pytest.approx(0.20 * 67_430.0)
    assert out["employee_national_insurance"] != pytest.approx(0.08 * 67_430.0)
    assert out["marginal_deduction_rate"] != pytest.approx(0.28)


def test_earnings_between_the_higher_rate_threshold_and_the_uel() -> None:
    """Higher-rate income tax but still main-rate NI: the bands are separate."""
    out = block(
        representative_earnings=55_000.0, ni_upper_earnings_limit_annual=62_000.0
    )
    assert out["marginal_deduction_rate"] == pytest.approx(0.40 + 0.08)
    assert out["employee_national_insurance"] == pytest.approx(
        0.08 * (55_000.0 - 12_570.0)
    )
    assert out["income_tax_band"] == "higher-rate"
    assert out["national_insurance_band"] == "main-rate"


def test_the_higher_rate_threshold_parameter_is_actually_read() -> None:
    """It used to be stored and never consulted, i.e. a guard on nothing."""
    narrow = block(
        representative_earnings=40_000.0,
        basic_rate_limit=10_000.0,
        higher_rate_threshold=22_570.0,
    )
    assert narrow["marginal_deduction_rate"] == pytest.approx(0.40 + 0.08)
    assert narrow["income_tax"] == pytest.approx(
        0.20 * 10_000.0 + 0.40 * (40_000.0 - 12_570.0 - 10_000.0)
    )
    wide = block(representative_earnings=40_000.0)
    assert wide["marginal_deduction_rate"] == pytest.approx(0.28)
    assert narrow["implied_gap_percentage_points"] != pytest.approx(
        wide["implied_gap_percentage_points"]
    )


def test_inconsistent_higher_rate_threshold_is_rejected() -> None:
    """Gross-earnings and taxable-income conventions must not be mixed."""
    with pytest.raises(RuntimeError, match="internally inconsistent"):
        block(higher_rate_threshold=37_700.0)


# ---------------------------------------------------------------------------
# Hard failures: missing parameters, out-of-range earnings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", SCHEDULE_REQUIRED_PARAMETERS)
def test_every_required_parameter_is_required(field: str) -> None:
    params = {k: v for k, v in BASE.items() if k != field}
    with pytest.raises(RuntimeError, match="missing statutory parameters"):
        schedule_benchmark_block({"parameters": params})


def test_earnings_below_the_allowance_hard_fail() -> None:
    """Both rates would be zero and the reported contrast degenerate."""
    with pytest.raises(RuntimeError, match="outside the range"):
        block(representative_earnings=10_000.0)


def test_earnings_below_both_thresholds_hard_fail() -> None:
    with pytest.raises(RuntimeError, match="outside the range"):
        block(representative_earnings=8_000.0, ni_primary_threshold_annual=9_000.0)


def test_earnings_in_the_allowance_taper_hard_fail() -> None:
    """Above 100,000 the true marginal rate is 60 per cent; unmodelled."""
    with pytest.raises(RuntimeError, match="outside the range"):
        block(representative_earnings=PERSONAL_ALLOWANCE_TAPER_THRESHOLD + 1.0)
    with pytest.raises(RuntimeError, match="outside the range"):
        block(representative_earnings=150_000.0)


def test_the_reported_validity_range_is_the_one_enforced() -> None:
    out = block()
    low, high = out["parameters"]["valid_earnings_range"]
    assert low == min(BASE["personal_allowance"], BASE["ni_primary_threshold_annual"])
    assert high == PERSONAL_ALLOWANCE_TAPER_THRESHOLD
    assert low < EXPOSED_MEAN_EARNINGS < high


def test_the_marginal_rate_must_exceed_the_average_rate() -> None:
    """The block's entire claim. A flat schedule must raise, not emit zero.

    With no personal allowance and no primary threshold the schedule is
    proportional: marginal equals average and \\ScheduleImpliedGap would be
    exactly zero, contradicting the mechanism the manuscript rests on.
    """
    with pytest.raises(RuntimeError, match="does not exceed the average"):
        block(
            representative_earnings=30_000.0,
            personal_allowance=0.0,
            ni_primary_threshold_annual=0.0,
            higher_rate_threshold=37_700.0,
        )


# ---------------------------------------------------------------------------
# The two representative-worker blocks share one schedule
# ---------------------------------------------------------------------------


def test_schedule_and_monthly_blocks_cannot_disagree() -> None:
    monthly = monthly_uc_block()
    p = monthly["parameters"]
    sched = schedule_benchmark_block(monthly)
    e = p["representative_earnings"]
    assert sched["income_tax"] == pytest.approx(income_tax(e, p))
    assert sched["employee_national_insurance"] == pytest.approx(employee_ni(e, p))
    # the 12-month spell zeroes the year, so its relief IS the annual liability
    twelve = monthly["spells"]["12m"]
    assert twelve["tax_relief_monthly_correct"] == pytest.approx(sched["income_tax"])
    assert twelve["ni_relief_monthly_correct"] == pytest.approx(
        sched["employee_national_insurance"]
    )


def test_monthly_block_ni_respects_the_upper_earnings_limit() -> None:
    """The monthly block's NI was also unbounded above; it is not now."""
    p = dict(BASE)
    assert employee_ni(80_000.0, p) == pytest.approx(
        0.08 * 37_700.0 + 0.02 * 29_730.0
    )
    assert employee_ni(80_000.0, p) != pytest.approx(0.08 * (80_000.0 - 12_570.0))


# ---------------------------------------------------------------------------
# Periodicity guards (the JSA rate's guard, applied to every periodic node)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "good,rng,periodicity",
    [
        (241.73, NI_PRIMARY_THRESHOLD_WEEKLY_RANGE, "weekly"),
        (967.0, NI_UPPER_EARNINGS_LIMIT_WEEKLY_RANGE, "weekly"),
        (424.90, UC_STANDARD_ALLOWANCE_MONTHLY_RANGE, "monthly"),
    ],
)
def test_periodicity_ranges_exclude_every_other_periodicity(
    good: float, rng: tuple[float, float], periodicity: str
) -> None:
    low, high = rng
    assert low < good < high
    assert _check_periodic_amount(
        good,
        where="test",
        what="test node",
        periodicity=periodicity,
        plausible_range=rng,
        used_as="tests",
    ) == good
    others = (
        (good * 52, good * 52 / 12, good / 7)
        if periodicity == "weekly"
        else (good * 12, good / 4.345, good / 30)
    )
    for wrong in (*others, 0.0, float("nan")):
        with pytest.raises(RuntimeError, match="periodicity"):
            _check_periodic_amount(
                wrong,
                where="test",
                what="test node",
                periodicity=periodicity,
                plausible_range=rng,
                used_as="tests",
            )


def test_monthly_block_rejects_a_wrongly_periodic_standard_allowance() -> None:
    """An annual standard allowance read as monthly is 12x too big."""
    params = {**BASE, "standard_allowance_single_25_plus_month": 424.90 * 12}
    with pytest.raises(RuntimeError, match="periodicity"):
        monthly_uc_block(params)


# ---------------------------------------------------------------------------
# The artifact carries the current note and a recorded parameter provenance
# ---------------------------------------------------------------------------


def test_the_artifact_note_is_the_current_module_constant() -> None:
    """The stale, INVERTED note shipped because the block never re-ran.

    `--only monthly` no longer needs policyengine-uk, so the note reaches the
    artifact from code on any machine.
    """
    stored = json.loads(Path("results/referee_fixes.json").read_text())
    assert stored["monthly_uc_bounding"]["notes"] == MONTHLY_UC_NOTES


def test_the_artifact_records_which_parameter_branch_ran() -> None:
    stored = json.loads(Path("results/referee_fixes.json").read_text())
    params = stored["monthly_uc_bounding"]["parameters"]
    assert params["parameter_source"] in PARAMETER_SOURCES
    assert params["parameter_source_options"] == list(PARAMETER_SOURCES)
    assert params["parameter_vintage"]
    for field in SCHEDULE_REQUIRED_PARAMETERS:
        assert field in params


def test_monthly_block_runs_without_policyengine() -> None:
    """The whole point of the fallback: the note can be regenerated anywhere."""
    out = monthly_uc_block()
    assert out["notes"] == MONTHLY_UC_NOTES
    assert out["parameters"]["parameter_source"] in PARAMETER_SOURCES
    assert set(out["spells"]) == {"3m", "6m", "12m"}
