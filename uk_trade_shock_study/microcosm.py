"""Microcosm UK data access, pinned through policyengine.py.

Every microsimulation in the study is built with
``policyengine.uk.managed_microsimulation()``, which resolves the certified
Microcosm UK 2024-25 build named in the policyengine.py release bundle
(``pyproject.toml`` pins ``policyengine[uk]``) and records its provenance on
``sim.policyengine_bundle``. The dataset lives in a private Hugging Face
repository, so a token with access must be available (``HUGGING_FACE_TOKEN``
or a saved ``huggingface-cli login``). The file is materialised once under
``data/`` relative to the working directory.

Microcosm UK carries survey inputs for a single period, its base year
(``BASE_YEAR`` = 2024, tax year 2024-25). Simulating an earlier rulebook
(the 2022-23 and 2023-24 energy-crisis instruments) needs those inputs
copied into the earlier period first; ``copy_inputs`` does that from the
dataset file itself, so every input column -- not only the ones a script
happens to read -- is present in the target period.
"""

from __future__ import annotations

import os
from functools import lru_cache
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numpy as np

#: The single period with survey inputs in the certified Microcosm UK build.
BASE_YEAR = 2024


def _ensure_token() -> None:
    """Expose a saved Hugging Face login to policyengine.py as an env var."""
    if os.environ.get("HUGGING_FACE_TOKEN"):
        return
    try:
        from huggingface_hub import get_token
    except ImportError:  # pragma: no cover
        return
    token = get_token()
    if token:
        os.environ["HUGGING_FACE_TOKEN"] = token


@lru_cache(maxsize=1)
def dataset_path() -> Path:
    """Local path of the certified Microcosm UK file (downloaded once)."""
    from policyengine.provenance.dataset_materialization import materialize_dataset

    _ensure_token()
    return Path(materialize_dataset("uk").path)


def microsimulation(reform=None):
    """A Microcosm UK ``Microsimulation`` pinned to the policyengine.py bundle."""
    import policyengine as pe

    _ensure_token()
    if reform is None:
        return pe.uk.managed_microsimulation()
    return pe.uk.managed_microsimulation(reform=reform)


#: Inputs the deflation never rescales: survey weights are a population
#: count, not a money amount, so the back-dated run keeps the same households
#: at the same grossing (a disclosed choice: the model's projection would
#: also shrink the weights by ONS population growth).
DEFLATION_EXCLUSIONS = frozenset({"household_weight"})

#: Loader renames: the simulation holds some dataset columns under another
#: name. Each alias is verified against the column's values before use.
_HOLDER_SUFFIXES = ("_before_lsr", "_reported", "_before_response")


@lru_cache(maxsize=4)
def _projection_ratios(years: tuple, source_year: int = BASE_YEAR) -> dict:
    """Per-record ratio (value at ``source_year``) / (value at year) for every
    dataset column PolicyEngine UK's own forward projection changes.

    No hand-made deflators: the certified file is relabelled as the earliest
    target year and projected forward to ``source_year`` with the model's own
    ``extend_single_year_dataset`` (the uprating_indices.yaml growth series
    plus its regional rent, council tax, water, property, savings-income and
    energy-price rules). Dividing a ``source_year`` value by the ratio
    therefore inverts exactly the projection the model would apply between
    the two years. Returns ``{year: {column: (table, source_values, ratio)}}``.
    """
    from policyengine_uk.data.dataset_schema import UKSingleYearDataset
    from policyengine_uk.data.economic_assumptions import extend_single_year_dataset
    from policyengine_uk.system import system

    first = min(years)
    if first >= source_year:
        raise ValueError("deflation targets must precede the source year")
    original = UKSingleYearDataset(str(dataset_path()))
    relabelled = original.copy()
    relabelled.time_period = str(first)
    chain = extend_single_year_dataset(
        relabelled, system.parameters, end_year=source_year
    ).datasets
    out = {}
    for year in years:
        cols = {}
        for table in original.table_names:
            top = getattr(chain[source_year], table)
            here = getattr(chain[year], table)
            raw = getattr(original, table)
            for col in raw.columns:
                if col in DEFLATION_EXCLUSIONS or top[col].dtype.kind != "f":
                    continue
                a = here[col].to_numpy(dtype=float)
                b = top[col].to_numpy(dtype=float)
                ok = np.isfinite(a) & np.isfinite(b) & (a != 0)
                ratio = np.ones_like(a)
                ratio[ok] = b[ok] / a[ok]
                if np.allclose(ratio, 1.0):
                    continue
                cols[col] = (table, raw[col].to_numpy(dtype=float), ratio)
        out[year] = cols
    return out


def _deflation_ratio(name: str, values: np.ndarray, ratios: dict):
    """The projection ratio for holder ``name``, or None if it is not a
    projected dataset column. The match is verified on the values."""
    candidates = [name] + [name[: -len(s)] for s in _HOLDER_SUFFIXES if name.endswith(s)]
    for col in candidates:
        if col not in ratios:
            continue
        _, raw, ratio = ratios[col]
        if len(raw) == len(values) and np.allclose(
            raw, np.asarray(values, dtype=float), rtol=1e-5, atol=1e-3, equal_nan=True
        ):
            return col, ratio
    return None


def copy_inputs(sim, years, source_year: int = BASE_YEAR, deflate: bool = False) -> int:
    """Copy every input held at ``source_year`` into each of ``years``.

    Values are taken from the simulation's own holders, not the raw file, so
    the loader's post-processing (renamed inputs such as
    ``employment_income`` -> ``employment_income_before_lsr``, encoded enums)
    carries over. Call before any calculation. Raises if any copy is rejected.

    ``deflate=False`` (the central design) copies levels unchanged: the
    target periods see 2024-25 nominal values (a disclosed limitation).
    ``deflate=True`` scales every input PolicyEngine UK's own dataset
    projection grows between the target year and ``source_year`` back by that
    projection's per-record factor (see ``_projection_ratios``); survey
    weights are held fixed. The variables scaled, with their mean factors,
    are recorded on ``sim._deflation``.
    """
    n, failed = 0, []
    targets = tuple(sorted(y for y in set(years) if y != source_year))
    ratios = _projection_ratios(targets, source_year) if deflate else {}
    report = {year: {} for year in targets} if deflate else None
    for name in sim.tax_benefit_system.variables:
        holder = sim.get_holder(name)
        known = [p for p in holder.get_known_periods() if str(p) == str(source_year)]
        if not known:
            continue
        values = holder.get_array(known[0])
        for year in years:
            if year == source_year:
                continue
            out = values
            if deflate:
                hit = _deflation_ratio(name, values, ratios[year])
                if hit is not None:
                    col, ratio = hit
                    out = (np.asarray(values, dtype=float) / ratio).astype(values.dtype)
                    report[year][name] = round(float(np.median(ratio[ratio != 1.0])), 4)
            try:
                sim.set_input(name, year, out)
                n += 1
            except Exception as exc:  # noqa: BLE001 - re-raised below
                failed.append((name, year, str(exc)[:80]))
    if failed:
        raise RuntimeError(f"{len(failed)} input copies failed: {failed[:5]}")
    if n == 0:
        raise RuntimeError(f"no inputs found at {source_year} to copy")
    if deflate:
        for year in targets:
            missed = sorted(
                col for col in ratios[year]
                if not any(col == k or k.startswith(col + "_") for k in report[year])
            )
            report[year] = {"scaled": report[year], "projected_columns_not_scaled": missed}
        sim._deflation = report
    sim._copied_inputs = n
    return n


def source_household_ids() -> np.ndarray:
    """Survey household each Microcosm household record was cloned from.

    Microcosm UK expands each source FRS household into several records, so
    sampling uncertainty must be resampled at the source-household level.
    """
    import pandas as pd

    frame = pd.read_hdf(dataset_path(), "household")
    return frame["source_household_id"].to_numpy()


def provenance(sim) -> dict:
    """Model and data release pins recorded on a managed simulation."""
    bundle = getattr(sim, "policyengine_bundle", None) or {}
    keys = (
        "bundle_id",
        "policyengine_version",
        "model_version",
        "default_dataset",
        "certified_data_build_id",
        "certified_data_artifact_sha256",
    )
    out = {k: bundle.get(k) for k in keys}
    out["policyengine_version"] = out["policyengine_version"] or _pkg_version("policyengine")
    out["model_version"] = out["model_version"] or _pkg_version("policyengine-uk")
    return out


def household_array(sim, name: str, period: int) -> np.ndarray:
    """Household-level values of ``name``; raises if the variable is absent."""
    if name not in sim.tax_benefit_system.variables:
        raise KeyError(f"variable not in this PolicyEngine UK build: {name}")
    return np.asarray(sim.calculate(name, period, map_to="household"))
