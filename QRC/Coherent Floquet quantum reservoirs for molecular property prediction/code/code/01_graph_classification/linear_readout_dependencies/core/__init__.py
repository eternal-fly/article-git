"""Shared implementation for the FINAL linear-readout experiments."""

from .crc3_baseline import (
    CRC3_Z_ZZ_METHOD,
    CRC3_Z_ZZ_PROTOCOL,
    compute_crc3_state,
    compute_crc3_z_zz_features,
    crc3_z_zz_dim,
    z_zz_readout,
)
from .esn_baseline import (
    ESN_STANDARD_METHOD,
    ESN_STANDARD_PROTOCOL,
    compute_esn_standard_features,
    esn_standard_dim,
)
from .linear_readout import (
    DEFAULT_SEEDS,
    ExperimentConfig,
    collect_scan_results,
    plot_fixed_q_scan,
    plot_q_equal_t_scan,
    prepare_fixed_q_layout,
    prepare_q_equal_t_layout,
    run_scan,
)

__all__ = [
    "CRC3_Z_ZZ_METHOD",
    "CRC3_Z_ZZ_PROTOCOL",
    "compute_crc3_state",
    "compute_crc3_z_zz_features",
    "crc3_z_zz_dim",
    "z_zz_readout",
    "ESN_STANDARD_METHOD",
    "ESN_STANDARD_PROTOCOL",
    "compute_esn_standard_features",
    "esn_standard_dim",
    "DEFAULT_SEEDS",
    "ExperimentConfig",
    "collect_scan_results",
    "plot_fixed_q_scan",
    "plot_q_equal_t_scan",
    "prepare_fixed_q_layout",
    "prepare_q_equal_t_layout",
    "run_scan",
]
