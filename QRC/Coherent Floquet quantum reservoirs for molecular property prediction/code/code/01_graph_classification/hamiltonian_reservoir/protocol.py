"""Frozen, pointwise-fair Hamiltonian-setting migration protocol.

This module is deliberately declarative.  It is the single source of truth for
the experiment grid, and it fails if a main-comparison row changes Q, T, input
coordinates, split, seed, or decoder between reservoir families.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
FINAL_ROOT = ROOT.parent
WORKSPACE = FINAL_ROOT.parents[1]
SOURCE_MIGRATION = FINAL_ROOT / "16_dissipative_protocol_migration"
HAMILTONIAN_BACKEND = WORKSPACE / "spatrace_dtc_qrc" / "01_algorithm_validation"

PROTOCOL_VERSION = "hamiltonian-setting-fair-open-closed-esn-six-seed-v1"
INJECTION_PROTOCOL = "direct_xyz_hamiltonian_exponential_no_microsteps_v1"
CHANNEL_PROTOCOL = "eventwise_reset_to_zero_exact_suffix_mixture_v1"
OBSERVABLES = "Z+all_ZZ"

DATASETS = ("BACE", "BBBP")
TOKEN_VARIANTS = ("canonical_general", "canonical_global_prior")
PRIMARY_TOKEN_VARIANT = "canonical_general"
SEEDS = (0, 10, 20, 30, 40, 50)

G = 0.84
W = 0.50
LAYERS = 3
FEATURES_PER_QUBIT = 3
CLOSED_P = 0.0
OPEN_P = 0.10

Q_VALUES = (12, 16)
T_VALUES = (2, 4, 6, 8, 10, 12, 16, 20, 24, 30)
JOINT_QT_VALUES = tuple(range(1, 21))
FIXED_SIZE_POINTS = ((12, 12), (16, 16))
MEMORY_Q = 12
MEMORY_T = 20
MEMORY_LAGS = (1, 2, 4, 6, 8, 12, 16)

DECODERS = ("linear", "rbf_svr")
PRIMARY_DECODER = "linear"
LINEAR_EPOCHS = 500
LINEAR_LEARNING_RATE = 1e-3
LINEAR_BATCH_SIZE = 32
SVR_C_VALUES = (0.1, 1.0, 10.0)
SVR_GAMMA_SCALES = (0.25, 1.0, 4.0)
SVR_EPSILON = 0.01
MATCHED_ESN_WIDTH = {q: q * (q + 1) // 2 for q in Q_VALUES}
WIDE_ESN_WIDTHS = (500, 1000, 2000)
WIDE_ESN_T_VALUES = (12, 30)
ESN_SPECTRAL_RADIUS = 0.90
ESN_INPUT_SCALE = 0.75
ESN_LEAK_RATE = 0.65

# A diagnostic scan only.  It never supplies a per-T, per-seed, or per-dataset
# p choice to the main comparison.  The dense low-p region resolves the
# expected transition around the prespecified OPEN_P=.10.
PT_Q = 12
PT_T_VALUES = (6, 12, 20, 30)
PT_P_VALUES = (
    0.00, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10,
    0.12, 0.16, 0.20, 0.30, 0.40, 0.60, 0.80,
)

# The strict 0.1 grid is shown as a heat map.  The canonical g=.84 point is
# evaluated separately because inserting it into the grid would make the cell
# spacing nonuniform.
WG_VALUES = tuple(round(index / 10, 1) for index in range(11))
WG_DATASET = "BACE"
WG_TOKEN_VARIANT = "canonical_general"
WG_Q = 12
WG_T = 30

OBSERVABLE_Q = 16
OBSERVABLE_T_VALUES = (6, 12, 16, 20)
OBSERVABLE_D_VALUES = (32, 64, 128)
OBSERVABLE_SUBSET_SEEDS = tuple(range(10))

CACHE_ROOT = ROOT / "cache"
RESULT_ROOT = ROOT / "results"
FIGURE_ROOT = ROOT / "figures"
MANIFEST_ROOT = ROOT / "manifests"


def readout_dim(q: int) -> int:
    q = int(q)
    return q + q * (q - 1) // 2


def source_token_path(dataset: str, variant: str) -> Path:
    return (
        SOURCE_MIGRATION
        / "cache"
        / "tokens"
        / f"{dataset.upper()}_{variant}_Q24_T30.npz"
    )


def source_split_path(dataset: str, seed: int) -> Path:
    return (
        SOURCE_MIGRATION
        / "cache"
        / "splits"
        / dataset.upper()
        / f"seed{int(seed):03d}.json"
    )


def qrc_bank_path(
    dataset: str,
    variant: str,
    q: int,
    t: int,
    seed: int,
    *,
    g: float = G,
    W_value: float = W,
) -> Path:
    parameter_tag = f"g{float(g):.3f}_W{float(W_value):.3f}"
    return (
        CACHE_ROOT
        / "qrc_suffix_banks"
        / dataset.upper()
        / variant
        / f"Q{int(q):02d}"
        / parameter_tag
        / f"T{int(t):02d}_seed{int(seed):03d}.npz"
    )


@dataclass(frozen=True)
class ComparisonJob:
    figure: str
    dataset: str
    token_variant: str
    q: int
    t: int
    seed: int
    decoder: str
    system: str
    width: int
    p: float | None
    g: float = G
    W: float = W
    layers: int = LAYERS
    primary: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _systems(q: int) -> tuple[tuple[str, int, float | None], ...]:
    width = readout_dim(q)
    return (
        ("closed_dtc_qrc", width, CLOSED_P),
        ("open_dtc_qrc", width, OPEN_P),
        ("matched_esn", width, None),
    )


def _pointwise_jobs(
    *, figure: str, dataset: str, variant: str, points: Iterable[tuple[int, int]],
    primary: bool = True,
) -> list[ComparisonJob]:
    rows: list[ComparisonJob] = []
    for q, t in points:
        for seed in SEEDS:
            for decoder in DECODERS:
                for system, width, p in _systems(q):
                    rows.append(
                        ComparisonJob(
                            figure=figure,
                            dataset=dataset,
                            token_variant=variant,
                            q=q,
                            t=t,
                            seed=seed,
                            decoder=decoder,
                            system=system,
                            width=width,
                            p=p,
                            primary=primary and decoder == PRIMARY_DECODER,
                        )
                    )
    return rows


def main_comparison_jobs() -> list[ComparisonJob]:
    """Rows underlying the migrated Fig. 3, Fig. 4, and Fig. 7 comparisons."""
    rows: list[ComparisonJob] = []
    # Fig. 3a is retained as a descriptive joint scaling scan.  It is never
    # interpreted as an isolated width or memory effect.
    rows += _pointwise_jobs(
        figure="fig3_joint_q_equals_t_descriptive",
        dataset="BACE",
        variant=PRIMARY_TOKEN_VARIANT,
        points=((value, value) for value in JOINT_QT_VALUES),
    )
    # Fig. 3b is the primary capacity-controlled molecular prefix comparison.
    rows += _pointwise_jobs(
        figure="fig3_fixed_q_prefix_primary",
        dataset="BACE",
        variant=PRIMARY_TOKEN_VARIANT,
        points=((q, t) for q in Q_VALUES for t in T_VALUES),
    )
    # Fig. 4 is a derived view of these fixed operating points, but BBBP must
    # also be extracted.  Duplicate BACE rows are intentionally not emitted.
    rows += _pointwise_jobs(
        figure="fig4_fixed_size_primary",
        dataset="BBBP",
        variant=PRIMARY_TOKEN_VARIANT,
        points=FIXED_SIZE_POINTS,
    )
    # Global-prior chemistry is a mirrored supplement, not mixed into the
    # canonical-general primary figures.
    for dataset in DATASETS:
        rows += _pointwise_jobs(
            figure="supp_global_prior_fixed_q_prefix",
            dataset=dataset,
            variant="canonical_global_prior",
            points=((q, t) for q in Q_VALUES for t in T_VALUES),
            primary=False,
        )
    # Fig. 7 uses fixed Q and fixed total T; only cue separation changes.
    for family in ("delayed_motif", "distant_pair"):
        for lag in MEMORY_LAGS:
            rows += _pointwise_jobs(
                figure=f"fig7_{family}_lag{lag:02d}",
                dataset=family,
                variant="controlled_sparse_local",
                points=((MEMORY_Q, MEMORY_T),),
            )
    return rows


def wide_esn_jobs() -> list[ComparisonJob]:
    """Capacity scaling kept separate from the pointwise main comparison."""
    rows: list[ComparisonJob] = []
    for dataset in DATASETS:
        for variant in TOKEN_VARIANTS:
            for q in Q_VALUES:
                widths = (readout_dim(q), *WIDE_ESN_WIDTHS)
                for t in WIDE_ESN_T_VALUES:
                    for seed in SEEDS:
                        for decoder in DECODERS:
                            for width in widths:
                                rows.append(
                                    ComparisonJob(
                                        figure="fig_esn_width_scan_separate",
                                        dataset=dataset,
                                        token_variant=variant,
                                        q=q,
                                        t=t,
                                        seed=seed,
                                        decoder=decoder,
                                        system="esn_width_scan",
                                        width=width,
                                        p=None,
                                        primary=False,
                                    )
                                )
    return rows


def wg_jobs() -> list[ComparisonJob]:
    """Full interaction grid; every test cell is retained, never selected."""
    rows: list[ComparisonJob] = []
    for g in WG_VALUES:
        for W_value in WG_VALUES:
            for seed in SEEDS:
                for system, p in (
                    ("closed_dtc_qrc", CLOSED_P),
                    ("open_dtc_qrc", OPEN_P),
                ):
                    rows.append(
                        ComparisonJob(
                            figure="fig_w_g_heatmaps",
                            dataset=WG_DATASET,
                            token_variant=WG_TOKEN_VARIANT,
                            q=WG_Q,
                            t=WG_T,
                            seed=seed,
                            decoder=PRIMARY_DECODER,
                            system=system,
                            width=readout_dim(WG_Q),
                            p=p,
                            g=g,
                            W=W_value,
                            primary=True,
                        )
                    )
    return rows


def pt_jobs() -> list[ComparisonJob]:
    """Fixed-Q p-by-T diagnostic; all cells are reported without selection."""
    rows: list[ComparisonJob] = []
    for dataset in DATASETS:
        for t in PT_T_VALUES:
            for p in PT_P_VALUES:
                for seed in SEEDS:
                    for decoder in DECODERS:
                        rows.append(
                            ComparisonJob(
                                figure="fig_p_by_T_diagnostic",
                                dataset=dataset,
                                token_variant=PRIMARY_TOKEN_VARIANT,
                                q=PT_Q,
                                t=t,
                                seed=seed,
                                decoder=decoder,
                                system="dtc_qrc_p_scan",
                                width=readout_dim(PT_Q),
                                p=p,
                                primary=False,
                            )
                        )
    return rows


def validate_protocol() -> dict[str, int]:
    if OPEN_P <= 0 or OPEN_P >= 1:
        raise AssertionError("OPEN_P must define a genuinely open, nontrivial channel")
    if W != 0.5:
        raise AssertionError("canonical W must remain 0.5")
    if tuple(round(b - a, 10) for a, b in zip(WG_VALUES, WG_VALUES[1:])) != (0.1,) * 10:
        raise AssertionError("W/g heat-map grid must have exact 0.1 spacing")
    main = main_comparison_jobs()
    # Every main-comparison key must expose all three systems at the same Q/T.
    grouped: dict[tuple, set[str]] = {}
    for row in main:
        key = (
            row.figure, row.dataset, row.token_variant, row.q, row.t,
            row.seed, row.decoder,
        )
        grouped.setdefault(key, set()).add(row.system)
        if row.system == "matched_esn" and row.width != readout_dim(row.q):
            raise AssertionError(row)
        if row.system == "closed_dtc_qrc" and row.p != CLOSED_P:
            raise AssertionError(row)
        if row.system == "open_dtc_qrc" and row.p != OPEN_P:
            raise AssertionError(row)
    expected = {"closed_dtc_qrc", "open_dtc_qrc", "matched_esn"}
    bad = [key for key, systems in grouped.items() if systems != expected]
    if bad:
        raise AssertionError(f"unpaired main-comparison keys: {bad[:3]}")
    wide = wide_esn_jobs()
    if any(row.figure != "fig_esn_width_scan_separate" for row in wide):
        raise AssertionError("wide ESN rows leaked into a main figure")
    wg = wg_jobs()
    if {row.p for row in wg if row.system == "open_dtc_qrc"} != {OPEN_P}:
        raise AssertionError("open p varies in the W/g scan")
    pt = pt_jobs()
    expected_pt = (
        len(DATASETS) * len(PT_T_VALUES) * len(PT_P_VALUES)
        * len(SEEDS) * len(DECODERS)
    )
    if len(pt) != expected_pt or {row.q for row in pt} != {PT_Q}:
        raise AssertionError("invalid p-by-T diagnostic grid")
    if OPEN_P not in {row.p for row in pt} or CLOSED_P not in {row.p for row in pt}:
        raise AssertionError("p-by-T scan must contain both canonical channels")
    return {
        "main_comparison_rows": len(main),
        "wide_esn_rows": len(wide),
        "wg_rows": len(wg),
        "pt_rows": len(pt),
        "observable_cells_per_system_dataset": (
            len(OBSERVABLE_T_VALUES)
            * len(OBSERVABLE_D_VALUES)
            * len(OBSERVABLE_SUBSET_SEEDS)
            * len(SEEDS)
        ),
    }
