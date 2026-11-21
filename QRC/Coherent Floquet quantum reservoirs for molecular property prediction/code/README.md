# Coherent Floquet quantum reservoirs for molecular property prediction

Source code for molecular graph classification, ethene electronic-gap prediction, and the superconducting experiments on Quafu’s Baihua and Shenglian devices. Each numbered directory contains the research code for one part of the paper.

| Directory under `code/` | Contents |
| --- | --- |
| `01_graph_classification/` | BFS event encoding, quantum reservoir evolution, classical readouts, and BACE/BBBP classification |
| `02_gap_prediction/` | SHNITSEL A01 trajectory encoding and electronic-gap forecasting |
| `03_reset_scans/` | Graph and trajectory reset-probability scans |
| `04_floquet_drive_scan/` | Floquet drive scans and comparisons across decoder and readout settings |
| `05_dephasing/` | Dephasing simulations and comparisons with coherent evolution |
| `06_hardware_demonstration/` | Circuit preparation, Quafu acquisition notebooks, and local analysis of measured counts |
| `07_esn_width_scan/` | Echo-state network width sweeps |
| `08_figures_and_validation/` | Publication plotting code and checks against saved predictions and counts |

`code/shared/` supplies common implementations. The environment records and requirement files are in [`code/environment/`](code/environment/). See [`code/LICENSE_NOTICE.md`](code/LICENSE_NOTICE.md) for source and license information.

## Working with the code alone

Run commands from the directory containing this README. List the 11 workflow entry points in [`index/WORKFLOW_ENTRY_POINTS.csv`](index/WORKFLOW_ENTRY_POINTS.csv). To display a workflow's parameters using only Python, run

```sh
python code/01_graph_classification/classify_molecular_graphs.py --describe
```

The graph protocol and the two-qubit evolution can be checked with synthetic inputs. With the core environment installed, run

```sh
(
  cd code/01_graph_classification/hamiltonian_reservoir
  PYTHONPATH=.:../../shared/quantum_dynamics python -m pytest -q -p no:cacheprovider \
    tests/test_protocol.py tests/test_hamiltonian_qrc.py
)
```

To draw the architecture schematic (Fig. 1) with Matplotlib, NumPy, and pandas installed, run

```sh
python code/08_figures_and_validation/publication_figures/render_figures.py --only 1 --out runs/figures
```

To check the source and documentation against the release checksums on Linux, run

```sh
sha256sum --check --quiet index/SOURCE_SHA256SUMS
```

## Reproducing numerical results

The graph tasks use MoleculeNet BACE and BBBP. The gap forecasts use SHNITSEL A01 trajectories. The hardware analysis reads the measured Quafu counts. The separate reproducibility archive contains the prepared splits, numerical inputs, measurements, and published figure assets.

To use the paper’s archived inputs, copy `data/` and `figures/` from the `QRC_reproducibility/` directory in `Coherent_Floquet_quantum_reservoirs_for_molecular_property_prediction_reproducibility.zip` into this directory, next to `code/` and `index/`. The launchers use [`index/RUNTIME_PATHS.csv`](index/RUNTIME_PATHS.csv) to reconstruct the original experiment paths inside `runs/runtime/`. The figure index lists the inputs for each panel group. Figures 4, 5, and 10 use the archived vector panels in the publication build.

With both companion directories in place, the full input and source check is

```sh
python code/_support/verify_package.py --json
```

See the [workflow instructions](index/REPRODUCTION.md) for the simulation settings, dependencies, and commands for each task, and the [figure index](index/FIGURE_CODE_DATA_INDEX.csv) for the plotting sources.

