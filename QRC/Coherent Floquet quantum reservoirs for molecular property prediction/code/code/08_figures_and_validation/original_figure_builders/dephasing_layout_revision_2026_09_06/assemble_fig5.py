"""Arrange the completed k=8 and k=32 dephasing curves as Figure 5.

This is a layout export from archived source rows. It does not simulate quantum
states, refit a model, or change any point or uncertainty band.
"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/qrc_dephasing_layout_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
WORKSPACE = PAPER.parents[2]
K8 = PAPER / 'source_data/supplement/coherence_dephasing_fixed_architecture'
K32 = WORKSPACE / 'BFS-QRC/reviewer_supplement_20260905/figures'
OUT = PAPER / 'fig/fig5_dephasing_k8_k32'
SOURCE = PAPER / 'source_data/main/fig5_dephasing_k8_k32'


def load_panels():
    sources = [K8 / 'mechanism_bace_auc.csv', K8 / 'mechanism_hopping_ratio_summary.csv',
               K32 / '09a_k32_bace_dephasing_summary.csv', K32 / '09b_k32_hopping_dephasing_summary.csv']
    rows = []
    for panel, k, task, path in zip('abcd', [8, 8, 32, 32], ['BACE', 'SHNITSEL', 'BACE', 'SHNITSEL'], sources):
        data = pd.read_csv(path)
        if 'readout' in data:
            data = data[data.readout == 'final']
        for row in data.itertuples():
            level = row.control_value if k == 8 else row.dephasing
            sd = row.sd if k == 8 else row.sd_across_seed_means
            low, high = (row.ci_low, row.ci_high) if k == 8 else (row.mean - sd, row.mean + sd)
            rows.append(dict(panel=panel, task=task, k=k, dephasing=level, p=row.p,
                             mean=row.mean, sd=sd, band_low=low, band_high=high,
                             band_type='95% bootstrap CI' if k == 8 else 'sample SD',
                             n_seeds=6, n_physical_test_trajectories=60 if task == 'SHNITSEL' else None,
                             source_file=str(path.relative_to(WORKSPACE)),
                             source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    data = pd.DataFrame(rows)
    assert len(data) == 24
    for _, block in data.groupby(['panel', 'p']):
        np.testing.assert_array_equal(sorted(block.dephasing), [0.0, 0.5, 1.0])

    # Check the new figure summaries against their existing per-seed source rows.
    for task, filename, metric in [('BACE', '09a_k32_bace_dephasing_source.csv', 'test_auc'),
                                   ('SHNITSEL', '09b_k32_hopping_dephasing_source.csv', 'mae_ratio_to_coherent')]:
        raw = pd.read_csv(K32 / filename)
        for (level, p), group in raw.groupby(['dephasing', 'p']):
            if task == 'SHNITSEL':
                assert len(group) == 6 * 60
            else:
                assert len(group) == 6
            values = group.groupby('seed')[metric].mean()
            assert list(values.index) == [0, 10, 20, 30, 40, 50]
            row = data[(data.task == task) & (data.k == 32) &
                       (data.dephasing == level) & (data.p == p)].iloc[0]
            np.testing.assert_allclose([values.mean(), values.std(ddof=1)], [row['mean'], row.sd], atol=1e-12)
    SOURCE.mkdir(parents=True, exist_ok=True)
    data.to_csv(SOURCE / 'plotted_data.csv', index=False)
    return data


def main():
    data = load_panels()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                         'axes.labelsize': 8.5, 'axes.titlesize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'axes.linewidth': 0.8, 'xtick.major.width': 0.8,
                         'ytick.major.width': 0.8, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none'})
    colors = {0.0: '#40578f', 0.1: '#c88a39'}
    markers = {0.0: 'o', 0.1: 's'}
    fig = plt.figure(figsize=(7.0, 6.4))
    axes = []
    for panel, (left, bottom) in zip('abcd', [(0.115, 0.59), (0.615, 0.59),
                                             (0.115, 0.14), (0.615, 0.14)]):
        ax = fig.add_axes([left, bottom, 0.32, 0.35])
        axes.append(ax)
        block = data[data.panel == panel]
        task, k = block.iloc[0][['task', 'k']]
        for p in [0.0, 0.1]:
            curve = block[block.p == p].sort_values('dephasing')
            ax.fill_between(curve.dephasing, curve.band_low, curve.band_high,
                            color=colors[p], alpha=0.14, lw=0, zorder=1)
            ax.plot(curve.dephasing, curve['mean'], color=colors[p], marker=markers[p],
                    markersize=4.0, lw=1.4, zorder=3)
        ax.set(xlim=(-0.05, 1.05), title=f'{task}, $k={k}$')
        ax.set_xticks([0, 0.5, 1], ['0.0', '0.5', '1.0'])
        if panel in 'cd':
            ax.set_xlabel(r'Event dephasing, $\lambda$', labelpad=4)
        if task == 'BACE':
            ax.set(ylim=(0.40, 0.85), ylabel='Test ROC-AUC')
            ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
            ref = 0.5
        else:
            ax.set(ylim=(0.97, 1.35), ylabel='Target-balanced MAE ratio')
            ax.set_yticks([1.0, 1.1, 1.2, 1.3])
            ref = 1.0
        assert block.band_low.min() > ax.get_ylim()[0]
        assert block.band_high.max() < ax.get_ylim()[1]
        ax.axhline(ref, color='#888888', ls='--', lw=0.8, zorder=2)
        ax.grid(axis='y', color='#999999', alpha=0.20, lw=0.6)
        ax.set_axisbelow(True)
        fig.text(left - 0.087, bottom + 0.35, panel, ha='left', va='bottom',
                 fontsize=10, fontweight='bold')
    handles = [Line2D([0], [0], color=colors[p], marker=markers[p], lw=1.4, ms=4,
                      label=('Closed, $p=0$' if p == 0 else 'Dissipative, $p=0.10$')) for p in [0.0, 0.1]]
    legend = fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
                        bbox_to_anchor=(0.52, 0.008), fontsize=8.5, columnspacing=2.0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    assert all(not legend_box.overlaps(ax.xaxis.label.get_window_extent(renderer)) for ax in axes[2:])
    for suffix in ['pdf', 'svg', 'png']:
        fig.savefig(OUT.with_suffix('.' + suffix), dpi=300, facecolor='white')
    plt.close(fig)
    report = dict(rows=len(data), panels=4, k8_series='original manuscript',
                  original_points_and_bands_preserved=True, k32_raw_seed_means_verified=True,
                  uncertainty={'a': 'paired-seed 95% bootstrap CI',
                               'b': 'crossed-seed/physical-trajectory 95% bootstrap CI',
                               'c': 'sample SD across six seed means',
                               'd': 'sample SD across six seed means'},
                  figure_size_inches=[7, 6.4], new_simulations=0, new_model_fits=0)
    (HERE / 'figure_data_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(data[['panel', 'k', 'dephasing', 'p', 'mean', 'band_type']].to_string(index=False))


if __name__ == '__main__':
    main()
