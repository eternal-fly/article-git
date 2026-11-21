"""Lay out the unchanged Figure 5 statistics at the final single-column size."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/qrc_fig5_single_column_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
spec = importlib.util.spec_from_file_location('archived_fig5', PAPER / 'dephasing_layout_revision_2026_09_06/assemble_fig5.py')
archived = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archived)


def main():
    data = archived.load_panels()
    source_name = 'source_data/main/fig5_dephasing_k8_k32/plotted_data.csv'
    assert (PAPER / source_name).read_bytes() == (HERE / 'before' / source_name).read_bytes()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.8,
                         'axes.labelsize': 6.8, 'axes.titlesize': 7.0,
                         'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                         'axes.linewidth': 0.65, 'xtick.major.width': 0.65,
                         'ytick.major.width': 0.65, 'xtick.major.size': 2.4,
                         'ytick.major.size': 2.4, 'xtick.major.pad': 2,
                         'ytick.major.pad': 2, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none'})
    width, height, side = 3.42, 3.64, 1.16
    fig = plt.figure(figsize=(width, height))
    axes = []
    panel_labels = []
    colors = {0.0: '#40578f', 0.1: '#c88a39'}
    markers = {0.0: 'o', 0.1: 's'}
    for panel, (left, bottom) in zip('abcd', [(0.40, 2.17), (2.06, 2.17),
                                             (0.40, 0.66), (2.06, 0.66)]):
        ax = fig.add_axes([left / width, bottom / height, side / width, side / height])
        axes.append(ax)
        block = data[data.panel == panel]
        task, k = block.iloc[0][['task', 'k']]
        for p in [0.0, 0.1]:
            curve = block[block.p == p].sort_values('dephasing')
            ax.fill_between(curve.dephasing, curve.band_low, curve.band_high,
                            color=colors[p], alpha=0.14, lw=0, zorder=1)
            ax.plot(curve.dephasing, curve['mean'], color=colors[p], marker=markers[p],
                    markersize=2.8, lw=1.0, zorder=3)
        ax.set(xlim=(-0.05, 1.05))
        ax.set_title(f'{task}, $k={k}$', pad=4)
        ax.set_xticks([0, 0.5, 1], ['0.0', '0.5', '1.0'])
        if panel in 'cd':
            ax.set_xlabel(r'Event dephasing, $\lambda$', labelpad=3)
        if task == 'BACE':
            ax.set(ylim=(0.40, 0.85))
            ax.set_ylabel('Test ROC-AUC', labelpad=3)
            ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
            reference = 0.5
        else:
            ax.set(ylim=(0.97, 1.35))
            ax.set_ylabel('Target-balanced\nMAE ratio', labelpad=3, linespacing=1.05)
            ax.set_yticks([1.0, 1.1, 1.2, 1.3])
            reference = 1.0
        assert block.band_low.min() > ax.get_ylim()[0]
        assert block.band_high.max() < ax.get_ylim()[1]
        ax.axhline(reference, color='#888888', ls='--', lw=0.6, zorder=2)
        ax.grid(axis='y', color='#999999', alpha=0.20, lw=0.45)
        ax.set_axisbelow(True)
        panel_labels.append(fig.text((left - 0.36) / width, (bottom + side + 0.08) / height,
                                     panel, ha='left', va='bottom', fontsize=8, fontweight='bold'))
    handles = [Line2D([0], [0], color=colors[p], marker=markers[p], lw=1, ms=2.8,
                      label=('Closed, $p=0$' if p == 0 else 'Dissipative, $p=0.10$')) for p in [0.0, 0.1]]
    legend = fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
                        bbox_to_anchor=(0.52, 0.012), fontsize=6.8, columnspacing=1.5,
                        handlelength=1.7, handletextpad=0.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_window_extent(renderer) for ax in axes]
    for box in boxes:
        np.testing.assert_allclose(box.width, box.height, atol=1e-9)
    legend_box = legend.get_window_extent(renderer)
    assert all(not legend_box.overlaps(ax.xaxis.label.get_window_extent(renderer)) for ax in axes[2:])
    for a, b in [(0, 1), (2, 3)]:
        assert not axes[a].get_tightbbox(renderer).overlaps(axes[b].get_tightbbox(renderer))
    for ax in axes:
        box = ax.get_tightbbox(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
    for label in panel_labels:
        box = label.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y1 <= fig.bbox.height
    out = PAPER / 'fig/fig5_dephasing_k8_k32'
    for suffix in ['pdf', 'svg', 'png']:
        fig.savefig(out.with_suffix('.' + suffix), dpi=300, facecolor='white')
    report = {'canvas_inches': [width, height], 'axis_side_inches': side,
              'layout': 'single-column 2x2 square axes', 'n_panels': len(axes),
              'n_source_rows': len(data), 'source_csv_byte_identical': True,
              'axis_pixel_dimensions': [[b.width, b.height] for b in boxes],
              'smallest_font_points': 6.5, 'panel_font_points': 8,
              'labels_inside_canvas': True, 'overlapping_panel_bounds': False,
              'legend_overlaps_xlabels': False, 'uncertainties_unchanged': True,
              'source_sha256': hashlib.sha256((PAPER / source_name).read_bytes()).hexdigest()}
    (HERE / 'figure5_qa.json').write_text(json.dumps(report, indent=2))
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
