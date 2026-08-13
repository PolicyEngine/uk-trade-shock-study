# Vintage-matched comparators for the LFS selection sensitivity

`results/lfs_selection_sensitivity.json` is built from the LFS five-quarter
longitudinal file (UKDA SN 9490), which is licensed separately from the FRS and
is not part of the Hugging Face download. It therefore could not be regenerated
in the 2026-08-13 re-run that corrected the Universal Credit award cache.

The quantity the manuscript quotes from that artifact is a **shift**: an
LFS-shaped displacement cushioning rate minus the uniformly-assigned one. The
cache fix moves both sides by roughly the same +2.7 points, so the shift is
robust to it — but only if both sides come from the same pipeline vintage.
Pairing the pre-fix LFS models against the re-run comparator gives
+0.07/+0.58/-0.78 instead of the correct +2.82/+3.33/+1.97: a three-point
error that reverses the finding.

These two files are the pre-fix comparators, preserved so the shift can be
computed within one vintage. They are NOT current estimates and must not be
quoted as levels anywhere. `analysis/write_lfs_selection_results.py` refuses to
run unless the comparator's vintage matches the LFS models', using the presence
of the `selection_method` field as the marker (it was added after these were
written, so pre-fix artifacts lack it and re-run artifacts carry it).

Delete this directory once the LFS sensitivity is re-run under the corrected
pipeline; the writer will then pair against `results/full_tariff_*.json`.
