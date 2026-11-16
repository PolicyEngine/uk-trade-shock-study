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


def copy_inputs(sim, years, source_year: int = BASE_YEAR) -> int:
    """Copy every input held at ``source_year`` into each of ``years``.

    Values are taken from the simulation's own holders, not the raw file, so
    the loader's post-processing (renamed inputs such as
    ``employment_income`` -> ``employment_income_before_lsr``, encoded enums)
    carries over. Call before any calculation. Levels are copied unchanged:
    the target periods see 2024-25 nominal values (a disclosed limitation).
    Raises if any copy is rejected.
    """
    n, failed = 0, []
    for name in sim.tax_benefit_system.variables:
        holder = sim.get_holder(name)
        known = [p for p in holder.get_known_periods() if str(p) == str(source_year)]
        if not known:
            continue
        values = holder.get_array(known[0])
        for year in years:
            if year == source_year:
                continue
            try:
                sim.set_input(name, year, values)
                n += 1
            except Exception as exc:  # noqa: BLE001 - re-raised below
                failed.append((name, year, str(exc)[:80]))
    if failed:
        raise RuntimeError(f"{len(failed)} input copies failed: {failed[:5]}")
    if n == 0:
        raise RuntimeError(f"no inputs found at {source_year} to copy")
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
