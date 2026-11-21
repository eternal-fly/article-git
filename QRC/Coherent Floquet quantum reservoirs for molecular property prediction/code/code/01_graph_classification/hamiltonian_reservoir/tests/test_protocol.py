from __future__ import annotations

from protocol import (
    CLOSED_P,
    OPEN_P,
    PT_P_VALUES,
    PT_Q,
    PT_T_VALUES,
    WG_VALUES,
    main_comparison_jobs,
    pt_jobs,
    readout_dim,
    validate_protocol,
    wg_jobs,
    wide_esn_jobs,
)


def test_protocol_is_pointwise_paired() -> None:
    counts = validate_protocol()
    assert counts["main_comparison_rows"] > 0
    assert OPEN_P == 0.10
    assert CLOSED_P == 0.0


def test_matched_esn_width_is_qrc_readout_width() -> None:
    for job in main_comparison_jobs():
        if job.system == "matched_esn":
            assert job.width == readout_dim(job.q)


def test_wide_esn_never_leaks_into_main_figure() -> None:
    assert not any(job.width in {500, 1000, 2000} for job in main_comparison_jobs())
    assert {500, 1000, 2000}.issubset({job.width for job in wide_esn_jobs()})


def test_wg_has_fixed_p_and_uniform_grid() -> None:
    assert WG_VALUES == tuple(round(index / 10, 1) for index in range(11))
    jobs = wg_jobs()
    assert {job.p for job in jobs if job.system == "closed_dtc_qrc"} == {0.0}
    assert {job.p for job in jobs if job.system == "open_dtc_qrc"} == {0.10}


def test_p_by_t_is_diagnostic_and_contains_canonical_points() -> None:
    jobs = pt_jobs()
    assert {job.q for job in jobs} == {PT_Q} == {12}
    assert {job.t for job in jobs} == set(PT_T_VALUES) == {6, 12, 20, 30}
    assert {job.p for job in jobs} == set(PT_P_VALUES)
    assert all(not job.primary for job in jobs)
