"""Frozen protocol for the six-seed open-DTC manuscript migration."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
FINAL_ROOT = ROOT.parent
WORKSPACE = FINAL_ROOT.parents[1]

PROTOCOL_VERSION = "canonical-v3-open-dtc-w0p5-six-seed-v1"
TOKEN_SCHEMA_VERSION = "local-bfs-canonical-isomeric-reparse-stable-v3"
GLOBAL_TOKEN_SCHEMA_VERSION = (
    "local-bfs-canonical-isomeric-reparse-stable-v3-global-prior-slots-v1"
)
CHANNEL_PROTOCOL = "eventwise_reset_to_zero_exact_suffix_mixture_v1"
OBSERVABLES = "Z+all_ZZ"

DATASETS = ("BACE", "BBBP")
TOKEN_VARIANTS = ("canonical_general", "canonical_global_prior")
SEEDS = (0, 10, 20, 30, 40, 50)
Q_VALUES = (12, 16)
T_VALUES = (2, 4, 6, 8, 10, 12, 16, 20, 24, 30)
P_VALUES = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.20, 0.40, 0.80)

G = 0.84
W = 0.50
LAYERS = 3
FEATURES_PER_QUBIT = 3
MAX_Q = 24
MAX_T = max(T_VALUES)

LINEAR_EPOCHS = 500
LINEAR_LR = 1e-3
LINEAR_BATCH_SIZE = 32
SELECTION_RULE = "max(val_auc) > min(val_loss) > max(val_acc)"

ESN_SPECTRAL_RADIUS = 0.90
ESN_INPUT_SCALE = 0.75
ESN_LEAK_RATE = 0.65
ESN_WIDE_WIDTHS = (500, 1000, 2000)

CACHE_ROOT = ROOT / "cache"
TOKEN_ROOT = CACHE_ROOT / "tokens"
QRC_FEATURE_ROOT = CACHE_ROOT / "qrc_suffix_banks"
ESN_FEATURE_ROOT = CACHE_ROOT / "esn"
SPLIT_ROOT = CACHE_ROOT / "splits"
RESULT_ROOT = ROOT / "results"
LOG_ROOT = ROOT / "logs"

LINEAR_ROOT = FINAL_ROOT / "2_linear_readout"
VALIDITY_ROOT = FINAL_ROOT / "5_graph_stream_validity"
QRC_BACKEND = LINEAR_ROOT / "core"


def readout_dim(q: int) -> int:
    q = int(q)
    return q + q * (q - 1) // 2


def esn_widths(q: int) -> tuple[int, ...]:
    return (readout_dim(q), *ESN_WIDE_WIDTHS)


def data_root(dataset: str) -> Path:
    return LINEAR_ROOT / "general" / dataset.upper() / "data"


def token_cache_path(dataset: str, variant: str) -> Path:
    return TOKEN_ROOT / f"{dataset.upper()}_{variant}_Q24_T{MAX_T:02d}.npz"


def token_metadata_path(dataset: str, variant: str) -> Path:
    return token_cache_path(dataset, variant).with_suffix(".json")


def qrc_bank_path(dataset: str, variant: str, q: int, t: int, seed: int) -> Path:
    return (
        QRC_FEATURE_ROOT
        / dataset.upper()
        / variant
        / f"Q{int(q):02d}"
        / f"T{int(t):02d}"
        / f"qrc_suffixes_seed{int(seed):03d}.npz"
    )


def esn_parameter_path(width: int, seed: int) -> Path:
    return ESN_FEATURE_ROOT / "parameters" / f"width{int(width):04d}_seed{int(seed):03d}.npz"


def esn_feature_path(
    dataset: str, variant: str, q: int, t: int, width: int, seed: int
) -> Path:
    return (
        ESN_FEATURE_ROOT
        / dataset.upper()
        / variant
        / f"Q{int(q):02d}"
        / f"T{int(t):02d}"
        / f"esn_width{int(width):04d}_seed{int(seed):03d}.npz"
    )


def split_path(dataset: str, seed: int) -> Path:
    return SPLIT_ROOT / dataset.upper() / f"seed{int(seed):03d}.json"
