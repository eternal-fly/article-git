# Reproducing the paper with the companion inputs

Run these commands from the repository root after placing the companion `data/` and `figures/` directories beside `code/` and `index/`. References below to supplied inputs, frozen features, measurements, and predictions refer to those companion files. For the source-only examples, see the [repository README](../README.md).

## Environment and execution paths

The recorded numerical environment uses Python 3.10.19, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, and PyTorch 2.10.0. Exact dependency records are in `code/environment/`. A fresh environment installation was not performed for this release.

```bash
python -m pip install -r code/environment/requirements-core.txt
python code/_support/verify_package.py --json
```

The CPU hopping backend also requires a C++17 compiler with OpenMP. GPU dephasing requires a compatible CUDA/PyTorch installation and Triton; see `code/environment/requirements-gpu.txt`. Regenerating graph tokens requires RDKit; its preprocessing version is recorded separately. Reading the supplied processed inputs does not require retokenization.

Each named workflow supports `--help` or `--describe` without preparing a runtime or importing research modules. Normal execution first copies the mapped files into `runs/runtime/`. Allow approximately 1 GiB of free space for preparation, plus space for new features and results. Large quantum runs need substantially more. To use a different scratch location, set `DTC_QRC_RUNTIME_DIR` to a new empty directory.

```bash
export DTC_QRC_RUNTIME_DIR=/path/to/scratch/dtc_qrc_run
```

The original code is executed inside that scratch tree, with its original relative imports. New graph and hopping entry-point results appear under `runs/runtime/runs/`; original research drivers use their usual output directories within the scratch tree. Released data and manuscript files are not used as experiment output locations. Internal implementation files are not standalone entry points in the categorized layout.

## Molecular graph classification

The main configuration uses Q=12, g=0.84, W=0.50, L=3, local chemical-prior BFS tokens, a final 78-dimensional Z+ZZ readout, and RBF-SVR. The supplied Q24 token arrays hold 72 coordinates; the Q12 application uses the first 36. Reset occurs after every graph event, including the last event. The application compares p=0 and the fixed p=0.10 setting.

```bash
python code/01_graph_classification/classify_molecular_graphs.py --dataset BACE --t 30 --seed 0
python code/01_graph_classification/classify_molecular_graphs.py --system esn --dataset BACE --t 30 --seed 0
```

The QRC entry defaults to supplied frozen features and refits the original decoder. `--features fresh --device cpu` requests quantum feature extraction. `--decoder linear` selects the original affine classification decoder instead of RBF-SVR. The datasets are BACE and BBBP; event lengths are 2, 4, 6, 8, 10, 12, 16, 20, 24, 30; seeds are 0, 10, 20, 30, 40, 50. The graph QRC readout width is fixed at 78. Standardization and decoder selection follow the supplied training and validation splits.

The graph output directory includes a digest of the requested configuration, so a two-point application run and a larger reset grid use distinct directories. This is a packaging-level output-path change; scientific functions are unchanged.

## Reset scans and ESN width

```bash
python code/03_reset_scans/scan_graph_reset_probability.py --dataset BACE --t 30 --seed 0
python code/03_reset_scans/scan_gap_reset_probability.py --seed 0 --threads 4
python code/07_esn_width_scan/scan_esn_width.py --dataset BBBP --t 30 --width 500 --seed 0
```

The graph scan uses T=6,12,20,30 and p=0,.01,.02,.04,.06,.08,.10,.12,.16,.20,.30,.40,.60,.80. Frozen feature files include these reset values at the scanned event lengths. The ESN width scan uses widths 78, 500, 1000, and 2000 at T=12 and 30. Graph controls reuse the original ESN input scaling, spectral radius, leak, and updates per event.

## Electronic-gap prediction and Floquet drive

SHNITSEL A01 provides 297 trajectories with eight windows per trajectory. The supplied processed array contains 2,376 windows, split by physical trajectory into 178/59/60 training/validation/test trajectories. Each window contains T=20 frames and 45 input coordinates for Q=15. Inputs encode distances and backward-difference velocities; targets are future electronic gaps and the minimum gap within the specified future interval. Reset acts between events, with no reset after the final event or during TM3 sampling.

```bash
python code/02_gap_prediction/predict_electronic_gaps.py --g 0.84 --seed 0 --threads 4
python code/04_floquet_drive_scan/scan_floquet_drive.py --g 0.88 --seed 0 --threads 4
```

The first command uses the main 27-p grid at fixed g=0.84. It includes the four targets, final/TM3 readouts, and Ridge/RBF-SVR decoders. In the manuscript, p is selected from the mean validation MAE across six seeds at fixed target/readout/decoder. Test MAE is not used for that selection.

The second command uses six fixed probabilities p=0,.2,.4,.6,.8,1. Each g/seed case yields 96 configurations. The complete saved scan spans g=0,.02,...,.98 and six seeds: 300 prediction archives and 28,800 metric rows. `--features-only` stops after quantum feature computation. The main-text scan emphasizes dissipative QRC with RBF-SVR and final readout; Appendix C compares both decoders and readouts. Relative-error and rank summaries describe the saved test scan, rather than redefining the fixed application setting.

The original GPU extractor and fixed-axis decoder evaluator are retained in `code/02_gap_prediction/surface_hopping/`. Their original runtime paths are recorded in `index/RUNTIME_PATHS.csv`. Earlier physical-selection results and off-grid shards are retained as supporting history; current manuscript rows are identified by the keys in the manuscript source tables.

## Dephasing

Figure 6 combines the original K=8 series with the full-dataset K=32 series. Original K=8 bands are confidence intervals; K=32 bands are sample standard deviations. Their identities are explicit in `figure06_dephasing_K8_K32.csv` and must be preserved when plotting.

```bash
python code/05_dephasing/run_original_K8_dephasing.py features --task bace --variant dephase_half --seed 0 --device cuda:0
python code/05_dephasing/run_full_dataset_dephasing.py --describe
```

The full-dataset source driver retains its original K=8,32,128 calculations. It is not a new K=32-only implementation. Its code, saved metric tables, backend verification record, and figure data are included. Large coherent suffix banks and noisy feature banks are not included; full quantum recomputation requires generating the coherent suffix banks first using the supplied graph/hopping extraction implementations. The original driver documents the required cache filenames in `coherent_bank_path`. The manuscript's original K=8 data should not be replaced with the separate K=8 output of this later driver.

## Hardware experiments

Raw counts, processed feature arrays, circuit-generation source, and acquisition notebooks are supplied for the Baihua and Shenglian chips. The hardware configuration uses Q=6, T=6, L=3, U3 inputs, W=0.6, and p=0. There are ten main/depth-control count archives and nine additional drive-scan archives; the g=0.84 drive reference uses the main Shenglian runs. Additional depth-repeat files remain explicitly separated.

```bash
python code/06_hardware_demonstration/analyze_hardware_correlations.py --describe
```

Omitting `--describe` invokes the original local correlation analysis and decoder fitting. It does not submit cloud jobs. Acquisition notebooks are source records and are not part of any automatic launch command. Raw ZZ features and connected correlations are distinct observable families; their tables are kept separately.

## Existing-result checks and figures

```bash
python code/08_figures_and_validation/check_saved_predictions_and_counts.py
python code/08_figures_and_validation/rebuild_manuscript_figures.py
```

The first command derives metrics and hardware features from saved predictions and counts. The second produces all ten current manuscript figures under `runs/runtime/runs/figures/`. The diagram uses the bundled Liberation Sans 2.1.5 faces so rebuilding on a host with several font versions preserves the final typography. It is drawn by `workflow.py`. The active plotting code uses the saved tables and the publication artwork recorded in the figure index. To redraw Figure 3 alone, run `python code/08_figures_and_validation/rebuild_manuscript_figures.py --only 3`. Its grouped bars and error bars use the six-seed test MAE means and sample standard deviations in `hopping_validation_selected_rbf_final.csv`.
