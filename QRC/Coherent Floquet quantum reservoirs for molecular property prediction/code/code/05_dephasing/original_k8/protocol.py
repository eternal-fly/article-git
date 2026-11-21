"""Frozen coherence and interaction ablations for the revised PR Applied paper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FINAL_ROOT = ROOT.parent
WORKSPACE = FINAL_ROOT.parents[1]

MAIN_ROOT = WORKSPACE / "BFS-QRC/FINAL/17_hamiltonian_setting_fair_migration"
SOURCE_ROOT = WORKSPACE / "BFS-QRC/FINAL/16_dissipative_protocol_migration"
HOPPING_ROOT = WORKSPACE / "Transition" / "shnitsel_a01_surface_hopping"
HAMILTONIAN_ROOT = WORKSPACE / "spatrace_dtc_qrc" / "01_algorithm_validation"

PROTOCOL_VERSION = "dtc-qrc-coherence-interaction-ablation-v2-six-seed-fixed-physical-axes"
INJECTION_PROTOCOL = "direct_xyz_hamiltonian_exponential_no_microsteps_v1"
OBSERVABLE_FAMILY = "Z+all_ZZ"
DECODER = "RBF-SVR"
BACE_SVR_C_VALUES = (0.1, 1.0, 10.0)
BACE_SVR_GAMMA_SCALES = (0.25, 1.0, 4.0)
BACE_SVR_EPSILON = 0.01

G = 0.84
W = 0.50
LAYERS = 3
P_VALUES = (0.0, 0.10)

BACE_Q = 12
BACE_T = 30
BACE_SEEDS = (0, 10, 20, 30, 40, 50)

HOPPING_Q = 15
HOPPING_T = 20
HOPPING_SEEDS = (0, 10, 20, 30, 40, 50)
HOPPING_ANGLE_SCALE = 0.50

DEPHASING_TRAJECTORIES = 8


@dataclass(frozen=True)
class Variant:
    name: str
    j_scale: float
    dephasing_strength: float
    trajectories: int


VARIANTS = (
    Variant("coherent", 1.0, 0.0, 1),
    Variant("interaction_half", 0.5, 0.0, 1),
    Variant("interaction_off", 0.0, 0.0, 1),
    Variant("dephase_half", 1.0, 0.5, DEPHASING_TRAJECTORIES),
    Variant("event_dephase", 1.0, 1.0, DEPHASING_TRAJECTORIES),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}


def feature_path(task: str, variant: str, seed: int) -> Path:
    return ROOT / "features" / task / variant / f"seed{int(seed):03d}.npz"


def bace_token_path() -> Path:
    return SOURCE_ROOT / "cache" / "tokens" / "BACE_canonical_general_Q24_T30.npz"


def bace_split_path(seed: int) -> Path:
    return SOURCE_ROOT / "cache" / "splits" / "BACE" / f"seed{int(seed):03d}.json"


def coherent_bace_path(seed: int) -> Path:
    return (
        MAIN_ROOT / "cache" / "qrc_suffix_banks" / "BACE"
        / "canonical_general" / "Q12" / "g0.840_W0.500"
        / f"T30_seed{int(seed):03d}.npz"
    )


def coherent_hopping_path(seed: int) -> Path:
    return HOPPING_ROOT / "features" / f"qrc_post_tm3_suffixes_seed{int(seed):03d}.npz"
