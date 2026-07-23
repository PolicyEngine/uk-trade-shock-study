from analysis.validate_manifest import validate


def test_frozen_manifest_matches_available_inputs():
    assert validate() == []
