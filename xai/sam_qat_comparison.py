'''
SAM maps side by side after quantisation
'''
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from scripts.dataloader import build_dataloaders
from xai.divergence_metrics import (RUNS_DIR, TOP_K, load_fp32_for_seed,
                                    load_quantised_run)
from xai.sam import (MANIFEST, compute_sam, denormalize_image,
                     get_layer_spikes, load_reference_model)
from xai.sam_metrics import samples_per_class, top_k_iou

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OUT_DIR = Path('results/Figures/Quantisation')
TABLES = Path('results/tables')
N_PER_CLASS = 20
K_SWEEP = (0.02, 0.05, 0.10, 0.20, 0.30)

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d8d7d2'
BOTH, ONLY_A, ONLY_B = '#3d3d3a', '#2a78d6', '#eb6834'
# Categorical slots validated for all pairs at four colours
CLASS_STYLE = (('#2a78d6', 'o', '-'), ('#eb6834', 's', '--'),
               ('#1baf7a', '^', ':'), ('#8b5fd6', 'D', '-.'))

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': None,
    'figure.facecolor': 'white', 'font.size': 9,
    'axes.titlesize': 9.5, 'text.color': INK, 'axes.labelcolor': INK,
    'axes.edgecolor': GRID, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'legend.frameon': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'lines.linewidth': 2,
})


def model_label(summary):
    '''A readable name: "membrane, 2 bit" rather than the arm slug.'''
    quant = summary.get('quantisation') or {}
    w, u = quant.get('weight_bits'), quant.get('membrane_bits')
    if w and u:
        return f'joint, {w} bit'
    if w:
        return f'weights, {w} bit'
    if u:
        return f'membrane, {u} bit'
    return 'control'


def summed_map(net, image, time_steps, device, layer_index, gamma):
    # SAM stack summed over time
    spikes = get_layer_spikes(net, image, time_steps, device, layer_index)
    return compute_sam(spikes, gamma).cpu().sum(dim=0)


def masks(map_a, map_b, k=TOP_K):
    #the two top-k masks, by the rule top_k_iou uses
    a, b = map_a.numpy(), map_b.numpy()
    return (a > np.quantile(a, 1.0 - k)), (b > np.quantile(b, 1.0 - k))


def pearson_of(map_a, map_b):
    a, b = map_a.flatten().numpy(), map_b.flatten().numpy()
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def chance_iou(k):
    # Overlap expected between two independent rankings at this k
    return k / (2 - k)

def draw_maps(chosen, classes, label, run, cmap):
    fig, axes = plt.subplots(len(classes), 4, figsize=(10.4, 2.65 * len(classes)))
    titles = ('spectrogram', 'SAM, full precision', f'SAM, {label}',
              f'top-{TOP_K:.0%} masks')
    mask_cmap = ListedColormap(['white', ONLY_B, ONLY_A, BOTH])

    for row, name in enumerate(classes):
        entry = chosen[name]
        a, b = entry['map_a'], entry['map_b']
        mask_a, mask_b = masks(a, b)
        panels = (denormalize_image(entry['image'], IMAGENET_MEAN, IMAGENET_STD)
                  .permute(1, 2, 0).numpy(),
                  (a / a.max()).numpy(), (b / b.max()).numpy(),
                  mask_a.astype(int) * 2 + mask_b.astype(int))

        for column, panel in enumerate(panels):
            ax = axes[row, column]
            ax.grid(False)
            if column == 0:
                ax.imshow(panel)
            elif column < 3:
                ax.imshow(panel, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.imshow(panel, cmap=mask_cmap, vmin=0, vmax=3,
                          interpolation='nearest')
                ax.set_title(f'IoU {entry["iou"]:.3f}', fontsize=9, color=INK)
            if row == 0 and column < 3:
                ax.set_title(titles[column])
            if column == 0:
                ax.set_ylabel(name.replace('_', ' '), fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_edgecolor(GRID)

    legend = [Patch(facecolor=BOTH, label='both'),
              Patch(facecolor=ONLY_A, label='full precision only'),
              Patch(facecolor=ONLY_B, label='quantised only')]
    fig.legend(handles=legend, loc='upper right', ncol=3, fontsize=8,
               bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(f'SAM maps of {run} against its full-precision parent',
                 x=0.008, ha='left', va='top', fontsize=11, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.5 / fig.get_figheight()])
    out = OUT_DIR / f'sam_comparison_{run}.png'
    fig.savefig(out)
    plt.close(fig)
    return out


def draw_sweep(sweep, classes, label, run):
    # Overlap against the size of the mask, raw and relative to chance
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.6, 4.4))
    height = fig.get_figheight()
    x = np.arange(len(K_SWEEP))
    reported = K_SWEEP.index(TOP_K) if TOP_K in K_SWEEP else None

    axl.plot(x, [chance_iou(k) for k in K_SWEEP], color=MUTED,
             linestyle=(0, (1, 2)), linewidth=1.4, label='chance, k / (2 − k)',
             zorder=2)
    axr.axhline(1.0, color=MUTED, linestyle=(0, (1, 2)), linewidth=1.4,
                zorder=2)
    span = len(classes) - 1
    for i, ((colour, marker, style), name) in enumerate(zip(CLASS_STYLE,
                                                            classes)):
        mean = np.array([sweep[name][k]['mean'] for k in K_SWEEP])
        axl.plot(x, mean, color=colour, marker=marker, linestyle=style,
                 label=name.replace('_', ' '), zorder=3)
        # Dodged, or the four ranges stack into one opaque bar at each k
        axl.vlines(x + 0.07 * (i - span / 2), [sweep[name][k]['min'] for k in K_SWEEP],
                   [sweep[name][k]['max'] for k in K_SWEEP], color=colour,
                   linewidth=0.9, alpha=0.3, zorder=2)
        axr.plot(x, mean / np.array([chance_iou(k) for k in K_SWEEP]),
                 color=colour, marker=marker, linestyle=style, zorder=3)

    for ax in (axl, axr):
        ax.set_xticks(x, [f'{k:.0%}' for k in K_SWEEP])
        ax.set_xlim(-0.2, len(K_SWEEP) - 0.8)
        ax.set_xlabel('k, fraction of pixels in the mask')
        if reported is not None:
            # The mask size every other table in the chapter reports
            ax.axvline(reported, color=MUTED, linewidth=0.8, alpha=0.45,
                       zorder=1)
    axl.set_ylim(0, 1)
    axl.set_ylabel('top-k overlap with the parent')
    axr.set_yscale('log')
    axr.set_ylim(0.8, 80)
    axr.set_yticks([1, 2, 5, 10, 20, 50], ['1×', '2×', '5×', '10×', '20×',
                                           '50×'])
    axr.set_ylabel('overlap divided by chance')
    if reported is not None:
        axr.annotate('reported k', (reported, 72), xytext=(-5, 0),
                     textcoords='offset points', ha='center', va='top',
                     rotation=90, fontsize=7.5, color=MUTED)

    handles, labels = axl.get_legend_handles_labels()
    order = list(range(1, len(labels))) + [0]
    fig.suptitle(f'Overlap against mask size, {label}', x=0.006,
                 y=1 - 0.06 / height, ha='left', va='top', fontsize=11,
                 weight='bold')
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc='upper left', bbox_to_anchor=(0.004, 1 - 0.30 / height),
               ncol=5, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(top=1 - 0.66 / height)
    out = OUT_DIR / f'sam_topk_sweep_{run}.png'
    fig.savefig(out)
    plt.close(fig)
    return out

def main():
    parser = argparse.ArgumentParser(
        description='SAM maps of a quantised model beside its full-precision '
                    'parent, and the overlap against the size of the mask')
    parser.add_argument('--run', type=str, default='qat_wxu2_112px_seed3456',
                        help='run directory name; the default is the seed '
                             'closest to its arm mean')
    parser.add_argument('--n-per-class', type=int, default=N_PER_CLASS)
    parser.add_argument('--cmap', type=str, default='magma',
                        help='sequential colour map for the SAM panels')
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run
    if not (run_dir / 'run.json').exists():
        raise SystemExit(f'no run.json under {run_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, time_steps, input_size, layer_index, gamma = load_reference_model(
        device, MANIFEST)

    quantised, handle, summary = load_quantised_run(run_dir, device)
    parent = load_fp32_for_seed(summary['seed'], input_size, device)
    label = model_label(summary)

    dataset, _, val_dataloader, _ = build_dataloaders(input_size=input_size,
                                                      verbose=False)
    classes = dataset.classes
    samples = samples_per_class(val_dataloader, len(classes), args.n_per_class)

    print(f'{args.run}  |  {label}  |  seed {summary["seed"]}  |  '
          f'{input_size} px  |  layer {layer_index}  |  gamma {gamma}')
    print(f'{args.n_per_class} validation images per class\n')

    rows, sweep_rows, chosen, sweep = [], [], {}, {}
    for position_label, images in samples.items():
        name = classes[position_label]
        per_image, per_k = [], {k: [] for k in K_SWEEP}
        for position, image in enumerate(images):
            a = summed_map(parent, image, time_steps, device, layer_index, gamma)
            b = summed_map(quantised, image, time_steps, device, layer_index, gamma)
            iou = top_k_iou(a, b, k=TOP_K)
            per_image.append({'iou': iou, 'position': position, 'image': image,
                              'map_a': a, 'map_b': b})
            row = {'run': args.run, 'class': name, 'position': position,
                   'iou': iou, 'pcc': pearson_of(a, b)}
            for k in K_SWEEP:
                value = top_k_iou(a, b, k=k)
                per_k[k].append(value)
                row[f'iou_k{k:g}'] = value
            rows.append(row)

        values = np.array([v['iou'] for v in per_image], dtype=float)
        median = float(np.nanmedian(values))
        pick = int(np.nanargmin(np.abs(values - median)))
        chosen[name] = per_image[pick]
        sweep[name] = {k: {'mean': float(np.nanmean(per_k[k])),
                           'min': float(np.nanmin(per_k[k])),
                           'max': float(np.nanmax(per_k[k]))} for k in K_SWEEP}
        for k in K_SWEEP:
            sweep_rows.append({'run': args.run, 'class': name, 'k': k,
                               'chance': chance_iou(k), **sweep[name][k]})
        print(f'{name:<18} median IoU {median:.3f}   chosen image '
              f'{per_image[pick]["position"]} with IoU {per_image[pick]["iou"]:.3f}'
              f'   interval [{np.nanmin(values):.3f}, {np.nanmax(values):.3f}]')
    handle.remove()

    print(f'\noverlap against mask size (mean over {args.n_per_class} images)')
    header = f'{"class":<18}' + ''.join(f'{f"k={k:.0%}":>12}' for k in K_SWEEP)
    print(header)
    print(f'{"chance":<18}' + ''.join(f'{chance_iou(k):>12.3f}' for k in K_SWEEP))
    print('-' * len(header))
    for name in classes:
        print(f'{name:<18}'
              + ''.join(f'{sweep[name][k]["mean"]:>12.3f}' for k in K_SWEEP))

    # The raw overlap is not comparable across k, because chance grows with the
    # mask: two masks covering a third of the map each overlap by construction
    print(f'\nratio to chance')
    print(header)
    print('-' * len(header))
    for name in classes:
        print(f'{name:<18}'
              + ''.join(f'{sweep[name][k]["mean"] / chance_iou(k):>11.1f}x'
                        for k in K_SWEEP))
    print(f'{"mean":<18}'
          + ''.join(f'{np.mean([sweep[n][k]["mean"] for n in classes]) / chance_iou(k):>11.1f}x'
                    for k in K_SWEEP))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    maps_png = draw_maps(chosen, classes, label, args.run, args.cmap)
    sweep_png = draw_sweep(sweep, classes, label, args.run)

    TABLES.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, data in ((f'sam_comparison_per_image_{args.run}', rows),
                       (f'sam_topk_sweep_{args.run}', sweep_rows)):
        path = TABLES / f'{stem}.csv'
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        written.append(path)

    print(f'\n{maps_png}\n{sweep_png}\n' + '\n'.join(str(w) for w in written))


if __name__ == '__main__':
    main()