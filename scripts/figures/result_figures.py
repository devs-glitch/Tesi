'''
Figures and tables for the quantisation study
'''
import csv
import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path('results')
FIGURES = RESULTS / 'Figures' / 'Quantisation'
TABLES = RESULTS / 'tables'

# Categorical slots 1-3 of the reference palette validated for all pairs
ARM_STYLE = {
    'weight-only':   ('#2a78d6', 'o', '-',  'weights only'),
    'membrane-only': ('#eb6834', 's', '--', 'membrane only'),
    'joint':         ('#1baf7a', '^', ':',  'joint'),
}
BITS = [8, 4, 2]
LAYERS = [1, 2, 3, 4]
DODGE = {'weight-only': -0.055, 'membrane-only': 0.0, 'joint': 0.055}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d8d7d2'
CAPTION_SIZE = 7.5

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': None,
    'figure.facecolor': 'white',
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'legend.frameon': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'lines.linewidth': 2, 'lines.markersize': 5,
})


def load(name):
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))


def arm_bits(run):
    # qat_w4ux_112px_seed1234 -> ('weight-only', 4)
    code = run.split('_')[1]
    w = None if code[1] == 'x' else int(code[1])
    u = None if code[3] == 'x' else int(code[3])
    arm = ('joint' if w and u else 'weight-only' if w else
           'membrane-only' if u else 'control')
    return arm, (w or u)


def seed_of(run):
    return int(re.search(r'seed(\d+)', run).group(1))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f'  {path}  ({len(rows)} rows)')


#  layout

def frame(fig, title=None, legend=None, legend_cols=3, hspace=None,
          xlabel=None, caption_text=None):
    height, width = fig.get_figheight(), fig.get_figwidth()
    head = 0.30 if title else 0.14

    lines, wrapped = 0, None
    if caption_text:
        wrapped = textwrap.fill(caption_text,
                                int(width * 72 / (CAPTION_SIZE * 0.58)))
        lines = wrapped.count('\n') + 1

    top = 1 - (head + (0.0 if legend is None else 0.26)) / height
    bottom = (0.08 + 0.152 * lines + (0.22 if xlabel else 0.0)) / height
    if title:
        fig.suptitle(title, x=0.008, y=1 - 0.06 / height, ha='left',
                     va='top', fontsize=11, weight='bold')
    if legend is not None:
        handles, labels = legend
        fig.legend(handles, labels, loc='upper left',
                   bbox_to_anchor=(0.006, 1 - head / height),
                   ncol=legend_cols, fontsize=8)
    fig.tight_layout(rect=[0, bottom, 1, top])
    if hspace is not None:
        fig.subplots_adjust(hspace=hspace)
    if wrapped:
        fig.text(0.008, 0.055 / height, wrapped, va='bottom',
                 fontsize=CAPTION_SIZE, color=MUTED, linespacing=1.45)
    if xlabel:
        # One label for the whole row drawn after the layout so it lands between the tick labels
        fig.text(0.5, bottom + 0.02 / height, xlabel, ha='center', va='bottom',
                 fontsize=9, color=INK)


def reference_line(ax, y, label=None, style=':', gutter=False, axis='y',
                   va=None):
    # reference levels are drawn under the data
    (ax.axhline if axis == 'y' else ax.axvline)(
        y, color=MUTED, linestyle=style, linewidth=1.1, zorder=1)
    if not label:
        return
    xy = (0.012 if gutter else 0.025, y) if axis == 'y' else (y, 0.26)
    coords = ('axes fraction', 'data') if axis == 'y' else ('data', 'axes fraction')
    ax.annotate(label, xy=xy, xycoords=coords, va=va or ('center' if axis == 'y' else 'bottom'),
                ha='left' if axis == 'y' else 'center', fontsize=7, color=MUTED,
                zorder=5, bbox=dict(facecolor='white', edgecolor='none', pad=1.2))


def reference_band(ax, low, high, mid, label=None, gutter=False):
    # An anchor measured over three seeds is a band
    ax.axhspan(low, high, color=MUTED, alpha=0.11, linewidth=0, zorder=1)
    ax.axhline(mid, color=MUTED, linestyle=(0, (6, 2, 1, 2)), linewidth=1.1,
               zorder=1)
    if label:
        ax.annotate(label, xy=(0.012 if gutter else 0.025, high),
                    xycoords=('axes fraction', 'data'), va='bottom', ha='left',
                    fontsize=7, color=MUTED, zorder=5,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.2))


def anchor_bands(fa):
    # (low, high, mid) for each metric, over the seeds of the control arm
    finals = [r['checkpoints']['final'] for r in fa['runs'].values()]
    out = {}
    for key, field in (('iou', 'mean_iou'), ('pcc', 'mean_pcc')):
        v = [f[field] for f in finals]
        out[key] = (min(v), max(v), float(np.mean(v)))
    return out


def spread(ax, x, low, high, colour):
    # minimum and maximum over the seeds
    ax.vlines(x, low, high, color=colour, linewidth=1.1, alpha=0.75, zorder=2)

# 1. Divergence of the explanation, per class, against bit width

def divergence_figure(div, classes, anchor=None):
    runs = div['runs']
    floor, chance = div['floor'], div['top_k'] / (2 - div['top_k'])

    rows = []
    for run, r in runs.items():
        arm, bits = arm_bits(run)
        for cls, m in r['divergence'].items():
            rows.append({'run': run, 'arm': arm, 'bits': bits,
                         'weight_bits': r['weight_bits'],
                         'membrane_bits': r['membrane_bits'],
                         'seed': r['seed'], 'class': cls,
                         'n_images': m['n'], 'pcc': m['pcc'],
                         'pcc_std': m['pcc_std'], 'iou': m['iou'],
                         # ssim is omitted: it is NaN in every record, because scikit-image was absent when divergence.py ran
                         'com_shift': m['com_shift'],
                         'com_shift_abs': m['com_shift_abs'],
                         'qat_accuracy': r['qat_accuracy']})
    write_csv(TABLES / 'divergence_per_class.csv', rows)

    def series(arm, cls, key):
        out = []
        for b in BITS:
            v = [x[key] for x in rows
                 if x['arm'] == arm and x['bits'] == b and x['class'] == cls]
            out.append((np.mean(v), min(v), max(v)))
        return np.array(out)

    fig, axes = plt.subplots(2, len(classes), figsize=(3.1 * len(classes), 6.1),
                             sharex=True, sharey='row')
    x = np.arange(len(BITS))

    for j, cls in enumerate(classes):
        for row, (key, lab) in enumerate((('iou', 'top-20 % overlap'),
                                          ('pcc', 'Pearson correlation'))):
            ax = axes[row, j]
            first = (j == 0)
            key_metric = 'iou' if row == 0 else 'pcc'
            if anchor:
                low, high, mid = anchor[key_metric]
                reference_band(ax, low, high, mid,
                               'fine-tuning alone' if first else None)
            # The floor label sits below its own line: the anchor band runs
            # just above it and the two would otherwise overlap.
            reference_line(ax, floor[key_metric],
                           'numerical floor' if first else None, '--', va='top')
            if row == 0:
                reference_line(ax, chance, 'chance level' if first else None)
            for arm, (c, mk, ls, name) in ARM_STYLE.items():
                s, xd = series(arm, cls, key), x + DODGE[arm]
                spread(ax, xd, s[:, 1], s[:, 2], c)
                ax.plot(xd, s[:, 0], color=c, marker=mk, linestyle=ls,
                        label=name if (row == 0 and first) else None, zorder=3)
            ax.set_xticks(x, [f'{b} bit' for b in BITS])
            ax.set_xlim(-0.42, len(BITS) - 0.58)
            if row == 0:
                ax.set_title(cls.replace('_', ' '))
            if first:
                ax.set_ylabel(lab)
    axes[0, 0].set_ylim(0, 0.98)
    axes[1, 0].set_ylim(0.70, 1.012)

    frame(fig, 'Explanation divergence from the full-precision model',
          legend=axes[0, 0].get_legend_handles_labels())

    fig.savefig(FIGURES / 'divergence_per_class.png')
    plt.close(fig)
    print(f'  {FIGURES / "divergence_per_class.png"}')

# 2. Faithfulness of the moved maps, per class

def faithfulness_figure(cf, classes):
    runs, thr = cf['runs'], cf['max_baseline_delta']
    rows = []
    for run, r in runs.items():
        for cls in classes:
            b, o, m = (r['baseline'][cls], r['own_maps'][cls],
                       r['reference_maps'][cls])
            rows.append({'run': run, 'arm': r['arm'], 'seed': r['seed'],
                         'class': cls, 'n_images': b['n'],
                         'baseline_delta': b['delta'], 'own_delta': o['delta'],
                         'reference_delta': m['delta'],
                         'baseline_auc_sam': b['auc_sam'],
                         'baseline_auc_random': b['auc_random'],
                         'own_auc_sam': o['auc_sam'],
                         'own_auc_random': o['auc_random'],
                         'reference_auc_sam': m['auc_sam'],
                         'reference_auc_random': m['auc_random'],
                         'has_signal': b['delta'] <= thr})
    write_csv(TABLES / 'faithfulness_per_class.csv', rows)

    seeds = sorted({r['seed'] for r in rows})
    fig, axes = plt.subplots(1, len(classes), figsize=(3.1 * len(classes), 3.9),
                             sharey=True)
    w = 0.26
    cols = [MUTED, '#eb6834', '#2a78d6']
    labels = ['FP32 baseline', 'own map', 'FP32 map on the 2-bit model']

    for j, cls in enumerate(classes):
        ax = axes[j]
        # A seed whose full-precision baseline shows no signal is shaded
        for i, s in enumerate(seeds):
            if not next(r['has_signal'] for r in rows
                        if r['seed'] == s and r['class'] == cls):
                ax.axvspan(i - 0.5, i + 0.5, color=GRID, alpha=0.55,
                           linewidth=0, zorder=0)
        ax.axhline(0, color=MUTED, linewidth=1, zorder=1)
        reference_line(ax, thr, 'interpretability threshold' if j == 0 else None,
                       '--')
        for k, key in enumerate(('baseline_delta', 'own_delta', 'reference_delta')):
            vals = [next(r[key] for r in rows if r['seed'] == s and r['class'] == cls)
                    for s in seeds]
            ax.bar(np.arange(len(seeds)) + (k - 1) * w, vals, w,
                   color=cols[k], label=labels[k] if j == 0 else None,
                   edgecolor='white', linewidth=0.8, zorder=3)
        ax.set_xticks(np.arange(len(seeds)), [f'seed {s}' for s in seeds],
                      fontsize=8)
        ax.set_xlim(-0.5, len(seeds) - 0.5)
        ax.set_title(cls.replace('_', ' '))
    axes[0].invert_yaxis()
    axes[0].set_ylabel('deletion Δ')

    frame(fig, 'Deletion-metric faithfulness, 2-bit membrane',
          legend=axes[0].get_legend_handles_labels())

    fig.savefig(FIGURES / 'faithfulness_per_class.png')
    plt.close(fig)
    print(f'  {FIGURES / "faithfulness_per_class.png"}')

# 3. Stratification by event strength, per class

def stratification_figure(strat, classes):
    runs, bins = strat['runs'], ['low', 'mid', 'high']
    rows = []
    for run, r in runs.items():
        arm, bits = arm_bits(run)
        for key, c in r['cells'].items():
            cls, b = key.split('|')
            rows.append({'run': run, 'arm': arm, 'membrane_bits': bits,
                         'seed': seed_of(run), 'class': cls,
                         'snr_bin': b, 'n_images': c['n'],
                         'median_snr': c['median_snr'], 'pcc': c['pcc'],
                         'iou': c['iou'], 'com_shift': c['com_shift']})
    write_csv(TABLES / 'stratification_per_class.csv', rows)

    seeds = sorted({r['seed'] for r in rows})
    # Colour encodes the bit width, as it encodes the arm in the other figures
    
    # the seed is a replicate and gets a thin line of the same colour
    depth = {2: ('#2a78d6', 'o', '-', '2-bit membrane'),
             4: (MUTED, 's', '--', '4-bit membrane')}

    fig, axes = plt.subplots(1, len(classes), figsize=(3.1 * len(classes), 3.7),
                             sharey=True)
    x = np.arange(len(bins))
    for j, cls in enumerate(classes):
        ax = axes[j]
        for mb, (c, mk, ls, name) in depth.items():
            per_seed = []
            for s in seeds:
                v = [next((r['iou'] for r in rows if r['seed'] == s
                           and r['class'] == cls and r['snr_bin'] == b
                           and r['membrane_bits'] == mb), np.nan) for b in bins]
                per_seed.append(v)
                ax.plot(x, v, color=c, linestyle=ls, linewidth=1.1, alpha=0.45,
                        zorder=2)
            ax.plot(x, np.nanmean(per_seed, axis=0), color=c, marker=mk,
                    linestyle=ls, label=name if j == 0 else None, zorder=3)
        ax.set_xticks(x, ['low', 'mid', 'high'])
        ax.set_xlim(-0.18, len(bins) - 0.82)
        ax.set_title(cls.replace('_', ' '))
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('top-20 % overlap')

    frame(fig, 'Explanation overlap by event strength',
          legend=axes[0].get_legend_handles_labels(), legend_cols=2,
          xlabel='SNR tercile within the class')
    fig.savefig(FIGURES / 'stratification_per_class.png')
    plt.close(fig)
    print(f'  {FIGURES / "stratification_per_class.png"}')

# 4. Behaviour against explanation: what actually moved

def behaviour_figure(grid_analysis, div, classes):
    runs = {k: v for k, v in grid_analysis['runs'].items()
            if arm_bits(k)[0] != 'control'}
    anchor = grid_analysis['independent_solutions']
    anchor_j = [p['predictions']['error_jaccard'] for p in anchor.values()]

    rows = []
    for run, r in runs.items():
        arm, bits = arm_bits(run)
        p, e = r['predictions'], r['emd']
        iou = [div['runs'][run]['divergence'][c]['iou'] for c in classes]
        rows.append({'run': run, 'arm': arm, 'bits': bits, 'seed': r['seed'],
                     'reference': r['reference'], 'accuracy': r['accuracy'],
                     'n': p['n'], 'reference_errors': p['reference_errors'],
                     'candidate_errors': p['candidate_errors'],
                     'shared_errors': p['shared_errors'],
                     'error_jaccard': p['error_jaccard'],
                     'predictions_differ': p['predictions_differ'],
                     'reference_right_candidate_wrong':
                         p['reference_right_candidate_wrong'],
                     'candidate_right_reference_wrong':
                         p['candidate_right_reference_wrong'],
                     'mean_iou': float(np.mean(iou)),
                     **{f'emd_layer{l}': e[f'layer{l}'] for l in LAYERS}})
    write_csv(TABLES / 'behaviour_per_run.csv', rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5),
                             gridspec_kw={'width_ratios': [1.3, 1]})

    # - left: the two divergences against each other one point per run
    ax = axes[0]
    ax.axhspan(min(anchor_j), max(anchor_j), color=MUTED, alpha=0.13,
               linewidth=0, zorder=1)
    reference_line(ax, float(np.mean(anchor_j)), 'independent solutions', '--')
    reference_line(ax, 1.0, 'identical decisions', '--')
    reference_line(ax, div['floor']['iou'], 'numerical floor', '--', axis='x')
    # Bit width is ordered, so it is encoded by marker area

    size = {8: 90, 4: 46, 2: 20}
    for arm, (c, mk, ls, name) in ARM_STYLE.items():
        s = [r for r in rows if r['arm'] == arm]
        ax.scatter([r['mean_iou'] for r in s], [r['error_jaccard'] for r in s],
                   c=c, marker=mk, s=[size[r['bits']] for r in s], label=name,
                   zorder=3, edgecolor='white', linewidth=0.7)
    arm_handles, arm_labels = ax.get_legend_handles_labels()
    key = [ax.scatter([], [], c=MUTED, marker='o', s=size[b],
                      label='_nolegend_') for b in BITS]
    ax.add_artist(ax.legend(handles=key, labels=[f'{b} bit' for b in BITS],
                            loc='lower right', fontsize=7.5, labelspacing=0.75,
                            borderpad=0.9, handletextpad=0.6))
    ax.set_xlabel('explanation preserved   (mean top-20 % overlap)')
    ax.set_ylabel('decision preserved\n(overlap of the two error sets)')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.05, 1.12)

    # - right: how far the firing-rate distribution moved, by layer
    ax = axes[1]
    x = np.arange(len(LAYERS))
    per_layer = [[p['emd'][f'layer{l}'] for p in anchor.values()] for l in LAYERS]
    ax.fill_between(x, [min(v) for v in per_layer], [max(v) for v in per_layer],
                    color=MUTED, alpha=0.13, linewidth=0, zorder=1)
    ax.plot(x, [np.mean(v) for v in per_layer], color=MUTED, linestyle='-.',
            linewidth=1.4, label='independent solutions', zorder=2)
    for arm, (c, mk, ls, name) in ARM_STYLE.items():
        s = [r for r in rows if r['arm'] == arm and r['bits'] == 2]
        v = [[r[f'emd_layer{l}'] for r in s] for l in LAYERS]
        xd = x + DODGE[arm]
        spread(ax, xd, [min(u) for u in v], [max(u) for u in v], c)
        ax.plot(xd, [np.mean(u) for u in v], color=c, marker=mk, linestyle=ls,
                zorder=3)
    ax.set_xticks(x, [f'layer {l}' for l in LAYERS])
    ax.set_xlim(-0.35, len(LAYERS) - 0.65)
    ax.set_ylabel('firing-rate shift\n(earth mover’s distance)')
    ax.annotate('2-bit arms', xy=(0.985, 0.93), xycoords='axes fraction',
                ha='right', fontsize=7.5, color=MUTED)

    handles = arm_handles + axes[1].get_legend_handles_labels()[0]
    labels = arm_labels + axes[1].get_legend_handles_labels()[1]
    frame(fig, 'What moved: the decision or the explanation',
          legend=(handles, labels), legend_cols=4)
    fig.savefig(FIGURES / 'behaviour_vs_explanation.png')
    plt.close(fig)
    print(f'  {FIGURES / "behaviour_vs_explanation.png"}')

# 5. Spike statistics per layer: regularisation or silence

def spike_figure():
    records = {}
    for path in sorted((RESULTS / 'grid').glob('*.json')):
        d = json.loads(path.read_text(encoding='utf-8'))
        records[d['tag']] = d

    rows = []
    for tag, d in records.items():
        arm, bits = ('fp32', None) if d['arm'] == 'fp32' else arm_bits(tag)
        for l in LAYERS:
            k = str(l)
            rows.append({'run': tag, 'arm': arm, 'bits': bits,
                         'seed': d['seed'], 'layer': l,
                         'n_neurons': d['n_neurons'][k],
                         'accuracy': d['accuracy'],
                         'mean_rate': d['mean_rate'][k],
                         'dead_fraction': d['dead_fraction'][k],
                         'saturated_fraction': d['saturated_fraction'][k],
                         'spike_com': d['spike_com'][k],
                         'cv_isi': d['cv_isi'][k],
                         'cv_isi_defined_fraction': d['cv_isi_defined_fraction'][k]})
    write_csv(TABLES / 'spike_statistics_per_layer.csv', rows)

    families = [('fp32', None, MUTED, 'D', '-.', 'full precision')]
    families += [(a, 2, c, mk, ls, f'{name}, 2 bit')
                 for a, (c, mk, ls, name) in ARM_STYLE.items()]

    panels = (('mean_rate', 'mean firing rate'),
              ('dead_fraction', 'silent neurons (fraction)'),
              ('cv_isi', 'CV of the inter-spike interval'),
              ('cv_isi_defined_fraction', 'positions with a defined CV'))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), sharex=True)
    x = np.arange(len(LAYERS))

    for ax, (key, lab) in zip(axes.flat, panels):
        for arm, bits, c, mk, ls, name in families:
            s = [r for r in rows if r['arm'] == arm and r['bits'] == bits]
            v = [[r[key] for r in s if r['layer'] == l] for l in LAYERS]
            xd = x + (0.0 if arm == 'fp32' else DODGE[arm])
            spread(ax, xd, [min(u) for u in v], [max(u) for u in v], c)
            ax.plot(xd, [np.mean(u) for u in v], color=c, marker=mk,
                    linestyle=ls, label=name if key == 'mean_rate' else None,
                    zorder=3)
        ax.set_ylabel(lab)
        ax.set_xticks(x, [f'layer {l}' for l in LAYERS])
        ax.set_xlim(-0.35, len(LAYERS) - 0.65)
    axes[1, 1].set_ylim(0.5, 1.03)

    frame(fig, 'Spike statistics per layer, at 2 bit',
          legend=axes[0, 0].get_legend_handles_labels(), legend_cols=4,
          hspace=0.22)
    fig.savefig(FIGURES / 'spike_statistics.png')
    plt.close(fig)
    print(f'  {FIGURES / "spike_statistics.png"}')

# 6. final figure: accuracy, explanation, energy against bit width

def summary_figure(div, energy, test, grid, classes, anchor=None):
    floor, chance = div['floor'], div['top_k'] / (2 - div['top_k'])
    res, bases = energy['results'], energy['baselines']

    rows = []
    for arm in ARM_STYLE:
        for b in BITS:
            runs = [t for t in res if t.startswith('qat')
                    and arm_bits(t) == (arm, b)]
            acc_t = [test['models'][t]['accuracy'] for t in runs]
            acc_v = [grid[t]['accuracy'] for t in runs]
            en = [res[t]['total_energy_pj'] / res[bases[t]]['total_energy_pj']
                  for t in runs]
            mem = [(res[t]['weight_bytes'] + res[t]['membrane_bytes'])
                   / (res[bases[t]]['weight_bytes'] + res[bases[t]]['membrane_bytes'])
                   for t in runs]
            row = {'arm': arm, 'bits': b, 'n_seeds': len(runs),
                   'test_accuracy_mean': np.mean(acc_t),
                   'test_accuracy_min': min(acc_t), 'test_accuracy_max': max(acc_t),
                   'val_accuracy_mean': np.mean(acc_v),
                   'energy_ratio_mean': np.mean(en),
                   'memory_ratio_mean': np.mean(mem)}
            for cls in classes:
                v = [div['runs'][t]['divergence'][cls]['iou'] for t in runs]
                row[f'iou_{cls}'] = np.mean(v)
                row[f'iou_{cls}_min'] = min(v)
                row[f'iou_{cls}_max'] = max(v)
            rows.append(row)
    write_csv(TABLES / 'summary_per_bit.csv', rows)

    fp32 = [test['models'][t]['accuracy'] for t in test['models']
            if t.startswith('fp32')]
    x = np.arange(len(BITS))
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.6), sharex=True)

    ax = axes[0]
    reference_line(ax, float(np.mean(fp32)), 'full precision', '--', gutter=True)
    for arm, (c, mk, ls, name) in ARM_STYLE.items():
        s, xd = [r for r in rows if r['arm'] == arm], x + DODGE[arm]
        spread(ax, xd, [r['test_accuracy_min'] for r in s],
               [r['test_accuracy_max'] for r in s], c)
        ax.plot(xd, [r['test_accuracy_mean'] for r in s], color=c, marker=mk,
                linestyle=ls, label=name, zorder=3)
    ax.set_ylabel('test accuracy')

    ax = axes[1]
    if anchor:
        low, high, mid = anchor['iou']
        reference_band(ax, low, high, mid, 'fine-tuning alone', gutter=True)
    reference_line(ax, floor['iou'], 'numerical floor', '--', gutter=True,
                   va='top')
    reference_line(ax, chance, 'chance level', gutter=True)
    for arm, (c, mk, ls, name) in ARM_STYLE.items():
        s, xd = [r for r in rows if r['arm'] == arm], x + DODGE[arm]
        for cls in classes:
            ax.plot(xd, [r[f'iou_{cls}'] for r in s], color=c, linestyle=ls,
                    alpha=0.4, linewidth=1.1, zorder=2)
        ax.plot(xd, [np.mean([r[f'iou_{cls}'] for cls in classes]) for r in s],
                color=c, marker=mk, linestyle=ls, zorder=3)
    ax.set_ylabel('explanation preserved\n(top-20 % overlap)')
    ax.set_ylim(0, 1)
    ax.annotate('thin lines: one per class', xy=(0.988, 0.055),
                xycoords='axes fraction', ha='right', fontsize=7.5, color=MUTED)

    ax = axes[2]
    reference_line(ax, 1.0, 'full precision', '--', gutter=True)
    for arm, (c, mk, ls, name) in ARM_STYLE.items():
        s, xd = [r for r in rows if r['arm'] == arm], x + DODGE[arm]
        ax.plot(xd, [r['energy_ratio_mean'] for r in s], color=c, marker=mk,
                linestyle=ls, zorder=3)
    ax.set_yscale('log')
    ax.set_ylabel('energy per inference\n(ratio, log scale)')
    ax.set_xticks(x, [f'{b} bit' for b in BITS])

    for ax in axes:
        ax.set_xlim(-0.60, len(BITS) - 0.70)

    frame(fig, 'Accuracy, explanation and energy against bit width',
          legend=axes[0].get_legend_handles_labels(), hspace=0.14)
    fig.savefig(FIGURES / 'summary_bits.png')
    plt.close(fig)
    print(f'  {FIGURES / "summary_bits.png"}')

def test_table(test):
    rows = []
    for name, v in test['models'].items():
        for cls, p in v['per_class'].items():
            rows.append({'model': name, 'arm': v['arm'],
                         'weight_bits': v['weight_bits'],
                         'membrane_bits': v['membrane_bits'], 'seed': v['seed'],
                         'class': cls, 'recall': p['recall'],
                         'support': p['support'],
                         'wilson_95_low': p['wilson_95'][0],
                         'wilson_95_high': p['wilson_95'][1]})
    write_csv(TABLES / 'test_recall_per_class.csv', rows)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    div = load('divergence.json')
    cf = load('cross_faithfulness.json')
    strat = load('stratified.json')
    energy = load('energy.json')
    grid_analysis = load('grid_analysis.json')
    anchor_path = RESULTS / 'finetuning_anchor.json'
    anchor = (anchor_bands(json.loads(anchor_path.read_text(encoding='utf-8')))
              if anchor_path.exists() else None)
    if anchor is None:
        print('  (no results/finetuning_anchor.json: the fine-tuning anchor '
              'will not be drawn)')
    test = json.loads((RESULTS / 'test' / 'test_evaluation.json').read_text(encoding='utf-8'))
    grid = {p.stem: json.loads(p.read_text(encoding='utf-8'))
            for p in (RESULTS / 'grid').glob('*.json')}
    classes = test['classes']

    print('tables and figures:')
    divergence_figure(div, classes, anchor)
    faithfulness_figure(cf, classes)
    stratification_figure(strat, classes)
    behaviour_figure(grid_analysis, div, classes)
    spike_figure()
    summary_figure(div, energy, test, grid, classes, anchor)
    test_table(test)
    print(f'\nfigures in {FIGURES}/, tables in {TABLES}/')


if __name__ == '__main__':
    main()