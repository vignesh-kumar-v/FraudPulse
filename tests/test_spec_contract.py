"""The feature spec is a contract. These tests are what make it one."""

from fraudpulse.features.spec import (
    ALL_MODEL_INPUTS,
    CATEGORICAL_FEATURE_NAMES,
    FEATURE_DEFAULTS,
    FEATURE_DEFS,
    FEATURE_DTYPES,
    FEATURE_NAMES,
    MODEL_STORE_FEATURE_NAMES,
    ONDEMAND_FEATURE_NAMES,
    STORE_ONLY_FEATURES,
    WINDOWS,
)


def test_no_duplicate_feature_names():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert len(ALL_MODEL_INPUTS) == len(set(ALL_MODEL_INPUTS))


def test_every_feature_has_a_dtype_a_default_and_a_description():
    for f in FEATURE_DEFS:
        assert f.dtype in {"int64", "float64"}, f.name
        assert f.description.strip(), f.name
        assert f.name in FEATURE_DEFAULTS and f.name in FEATURE_DTYPES


def test_store_only_features_are_stored_but_never_modelled():
    for name in STORE_ONLY_FEATURES:
        assert name in FEATURE_NAMES, f"{name} must exist in the store"
        assert name not in MODEL_STORE_FEATURE_NAMES
        assert name not in ALL_MODEL_INPUTS


def test_ondemand_features_are_not_stored():
    """If it were storable it would not be on-demand, and vice versa."""
    assert not set(ONDEMAND_FEATURE_NAMES) & set(FEATURE_NAMES)


def test_categoricals_are_in_the_model_inputs_but_not_the_store():
    for c in CATEGORICAL_FEATURE_NAMES:
        assert c in ALL_MODEL_INPUTS
        assert c not in FEATURE_NAMES


def test_every_window_produces_the_full_aggregate_set():
    for w in WINDOWS:
        for prefix in ("txn_count", "amt_sum", "amt_mean", "amt_max"):
            assert f"{prefix}_{w}" in FEATURE_NAMES


def test_windows_are_strictly_increasing():
    values = list(WINDOWS.values())
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_feast_definitions_import_and_match_the_spec():
    """The registry and the spec must not be able to drift apart."""
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1] / "feature_repo"
    sys.path.insert(0, str(repo))
    try:
        import definitions
    finally:
        sys.path.remove(str(repo))

    assert {f.name for f in definitions.card_stats.schema} == set(FEATURE_NAMES)
    assert {f.name for f in definitions.txn_ratios.features} == set(ONDEMAND_FEATURE_NAMES)
