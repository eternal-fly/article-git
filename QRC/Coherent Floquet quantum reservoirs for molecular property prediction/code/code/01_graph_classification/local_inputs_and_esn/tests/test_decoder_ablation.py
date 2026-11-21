from __future__ import annotations

import numpy as np
import torch

from decoder_ablation import (
    DECODERS,
    MLP_EPOCHS,
    MLP_WEIGHT_DECAYS,
    SVR_C,
    SVR_GAMMA_SCALES,
    RegularizedMLP,
    _score_metrics,
)
from run_decoder_q_t_scan import grid_case_rows
from run_bace_general_g_w_scan import (
    BASE_G,
    BASE_W,
    G_VALUES,
    W_VALUES,
    case_rows as g_w_case_rows,
    parameter_pairs,
)
from run_bace_general_g_w_heatmap import (
    G_VALUES as HEATMAP_G_VALUES,
    W_VALUES as HEATMAP_W_VALUES,
    case_rows as heatmap_case_rows,
    parameter_pairs as heatmap_parameter_pairs,
    validated_p_by_seed,
)


def test_decoder_search_contract():
    assert DECODERS == ("SVR_RBF", "REG_MLP")
    assert SVR_C == (0.1, 1.0, 10.0)
    assert SVR_GAMMA_SCALES == (0.25, 1.0, 4.0)
    assert MLP_WEIGHT_DECAYS == (1e-4, 1e-3, 1e-2)
    assert MLP_EPOCHS == 500


def test_regularized_mlp_preserves_batch_axis():
    model = RegularizedMLP(78).eval()
    with torch.no_grad():
        output = model(torch.zeros(5, 78))
    assert output.shape == (5,)
    assert model.hidden >= model.hidden2


def test_svr_metrics_keep_auc_score_but_clip_probability_metrics():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([-3.0, 0.25, 0.75, 4.0])
    metrics = _score_metrics(labels, scores)
    assert metrics["auc"] == 1.0
    assert metrics["acc"] == 1.0
    assert np.isclose(metrics["brier"], 0.03125)
    assert np.isclose(metrics["rmse"], np.sqrt(0.03125))


def test_full_decoder_grid_migrates_fixed_q_scanned_t_protocol():
    cases = grid_case_rows()
    assert len(cases) == 2880
    assert set(cases.q) == {12, 16}
    assert set(cases.t) == {2, 4, 6, 8, 10, 12, 16, 20, 24, 30}
    assert len(cases[cases.system != "ESN"]) == 960
    assert len(cases[cases.system == "ESN"]) == 1920


def test_bace_g_w_scan_is_paired_one_factor_plan():
    pairs = parameter_pairs()
    assert len(pairs) == len(G_VALUES) + len(W_VALUES) - 1 == 21
    assert len(set(pairs)) == len(pairs)
    assert pairs.count((BASE_G, BASE_W)) == 1
    assert len(g_w_case_rows("full")) == 126


def test_bace_g_w_heatmap_is_full_grid_with_frozen_validated_p():
    pairs = heatmap_parameter_pairs()
    assert HEATMAP_G_VALUES == tuple(round(i / 10, 1) for i in range(1, 11))
    assert HEATMAP_W_VALUES == tuple(round(i / 10, 1) for i in range(10))
    assert len(pairs) == 100
    assert len(set(pairs)) == 100
    assert len(heatmap_case_rows("full")) == 600
    assert validated_p_by_seed() == {
        0: 0.08,
        10: 0.0,
        20: 0.04,
        30: 0.08,
        40: 0.0,
        50: 0.02,
    }
