from protocol import HOPPING_SEEDS, P_VALUES, VARIANT_BY_NAME


def test_frozen_variant_grid() -> None:
    assert HOPPING_SEEDS == (0, 10, 20, 30, 40, 50)
    assert P_VALUES == (0.0, 0.10)
    assert (VARIANT_BY_NAME["coherent"].j_scale, VARIANT_BY_NAME["coherent"].dephasing_strength) == (1.0, 0.0)
    assert VARIANT_BY_NAME["interaction_off"].j_scale == 0.0
    assert VARIANT_BY_NAME["event_dephase"].dephasing_strength == 1.0
