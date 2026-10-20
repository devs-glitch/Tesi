'''
How much the dead and saturated fractions depend on where the cutoff is put
'''

import argparse
import json
import statistics as st
from pathlib import Path, PurePosixPath

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.dead_neurons_by_resolution import (DEAD_THRESHOLD, LAYERS,
                                        SATURATED_THRESHOLD,
                                        per_neuron_firing_rates)

RUNS_DIR = Path('runs')
MANIFEST = Path('config/baseline_manifest.json')
OUT_DIR = Path('results/Figures')
TABLES = Path('results/tables')
CACHE_NAME = 'per_neuron_rates.npz'

# Two decades around the adopted cutoff, and the saturation range that matters
DEAD_GRID = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10)
SAT_GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 0.99)

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d8d7d2'
LAYER_STYLE = (('#2a78d6', 'o', '-'), ('#eb6834', 's', '--'),
               ('#1baf7a', '^', ':'), ('#8b5fd6', 'D', '-.'))

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': None,
    'figure.facecolor': 'white', 'font.size': 9,
    'axes.titlesize': 9.5, 'text.color': INK, 'axes.labelcolor': INK,
    'axes.edgecolor': GRID, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'legend.frameon': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'lines.linewidth': 2,
})

def reference_runs():
    # The three runs the manifest points at
    
    with open(MANIFEST) as f:
        checkpoints = json.load(f)['checkpoints']
    return [RUNS_DIR / PurePosixPath(c.replace('\\', '/')).parent.name / 'run.json'
            for c in checkpoints]


def rates_for_run(run_json, device, recompute=False):
    # Per-neuron mean firing rates by layer
    cache = run_json.parent / CACHE_NAME
    if cache.exists() and not recompute:
        stored = np.load(cache)
        return {l: stored[f'layer{l}'] for l in LAYERS}, True

    from model.snn_model import GWGlitchSNN
    from scripts.dataloader import build_dataloaders

    with open(run_json) as f:
        summary = json.load(f)
    hp = summary['hyperparameters']
    _, _, val_dl, _ = build_dataloaders(input_size=summary['input_size'],
                                        verbose=False)
    net = GWGlitchSNN(beta=hp['beta'],
                      input_size=summary['input_size']).to(device)
    net.load_state_dict(torch.load(run_json.parent / 'best_model.pt',
                                   map_location=device))
    rates = per_neuron_firing_rates(net, val_dl, hp['time_steps'], device)
    rates = {l: rates[l].numpy() for l in LAYERS}

    np.savez_compressed(cache, **{f'layer{l}': rates[l] for l in LAYERS})
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rates, False


def sweep_one(rates):
    # Percentage below each dead cutoff and above each saturation cutoff
    out = {}
    for l in LAYERS:
        r = rates[l]
        out[l] = {
            'n_neurons': int(r.size),
            'max_rate': float(r.max()),
            'min_rate': float(r.min()),
            'dead': {t: float((r < t).mean() * 100) for t in DEAD_GRID},
            'saturated': {t: float((r > t).mean() * 100) for t in SAT_GRID},
        }
    return out


def aggregate(per_run, kind, grid):
    # Mean and standard deviation across seeds, per layer and threshold
    stats = {}
    for l in LAYERS:
        mean, sd = [], []
        for t in grid:
            values = [run[l][kind][t] for run in per_run]
            mean.append(st.mean(values))
            sd.append(st.stdev(values) if len(values) > 1 else 0.0)
        stats[l] = (np.array(mean), np.array(sd))
    return stats

def draw(dead_stats, sat_stats, label, out_dir):
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.4, 3.9))
    height = fig.get_figheight()

    for ax, grid, stats, adopted, xlabel in (
            (axl, DEAD_GRID, dead_stats, DEAD_THRESHOLD,
             'firing rate below which a neuron is dead'),
            (axr, SAT_GRID, sat_stats, SATURATED_THRESHOLD,
             'firing rate above which a neuron is saturated')):
        x = np.asarray(grid)
        for (colour, marker, style), l in zip(LAYER_STYLE, LAYERS):
            mean, sd = stats[l]
            ax.plot(x, mean, color=colour, marker=marker, linestyle=style,
                    markersize=5, label=f'layer {l}', zorder=3)
            ax.fill_between(x, mean - sd, mean + sd, color=colour, alpha=0.13,
                            linewidth=0, zorder=2)
        ax.axvline(adopted, color=MUTED, linewidth=1.0, alpha=0.5, zorder=1)
        ax.annotate(f'adopted, {adopted:g}', (adopted, 1.0),
                    xycoords=('data', 'axes fraction'), xytext=(-4, -4),
                    textcoords='offset points', ha='right', va='top',
                    rotation=90, fontsize=7, color=MUTED)
        ax.set_xlabel(xlabel, fontsize=7.6)
        ax.set_ylim(bottom=0)

    axl.set_xscale('log')
    axl.set_xticks(DEAD_GRID, [f'{t:g}' for t in DEAD_GRID], fontsize=7.5)
    axl.set_ylabel('neurons below the cutoff, %')
    axr.set_xticks(SAT_GRID, [f'{t:g}' for t in SAT_GRID], fontsize=7.5)
    axr.set_ylabel('neurons above the cutoff, %')
    # A flat zero is the result on the saturation side, so the axis has to show
    # that zero is zero rather than autoscaling onto numerical dust
    if max(float(sat_stats[l][0].max()) for l in LAYERS) < 1.0:
        axr.set_ylim(0, 1.0)

    handles, labels = axl.get_legend_handles_labels()
    fig.suptitle(f'Threshold sensitivity, {label}', x=0.006,
                 y=1 - 0.06 / height, ha='left', va='top', fontsize=10.5,
                 weight='bold')
    fig.legend(handles, labels, loc='upper left',
               bbox_to_anchor=(0.004, 1 - 0.30 / height), ncol=4, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(top=1 - 0.74 / height)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ('png', 'pdf'):
        path = out_dir / f'dead_neuron_sweep.{suffix}'
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    return written


def write_table(per_run, tags, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        f.write('run,layer,kind,threshold,percentage\n')
        for tag, run in zip(tags, per_run):
            for l in LAYERS:
                for kind, grid in (('dead', DEAD_GRID), ('saturated', SAT_GRID)):
                    for t in grid:
                        f.write(f'{tag},{l},{kind},{t},{run[l][kind][t]:.6f}\n')
    return path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='*', default=None,
                        help='folders in runs/ to analyse; the default is the '
                             'three checkpoints named by the manifest')
    parser.add_argument('--recompute', action='store_true',
                        help='ignore the cached per-neuron vectors')
    parser.add_argument('--label', default='three full-precision seeds at 112 px')
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.runs:
        run_jsons = [RUNS_DIR / r / 'run.json' for r in args.runs]
    else:
        run_jsons = reference_runs()
    run_jsons = [p for p in run_jsons if p.exists()]
    if not run_jsons:
        raise SystemExit('no run.json found; pass --runs')

    per_run, tags = [], []
    for rj in run_jsons:
        rates, cached = rates_for_run(rj, device, args.recompute)
        print(f'{rj.parent.name}: {"cache" if cached else "measured"}')
        per_run.append(sweep_one(rates))
        tags.append(rj.parent.name)

    dead_stats = aggregate(per_run, 'dead', DEAD_GRID)
    sat_stats = aggregate(per_run, 'saturated', SAT_GRID)

    print(f'\ndead fraction (%), mean over {len(per_run)} seeds')
    header = f'{"":<10}' + ''.join(f'{t:>9g}' for t in DEAD_GRID)
    print(header)
    for l in LAYERS:
        print(f'layer {l}   ' + ''.join(f'{v:>9.1f}' for v in dead_stats[l][0]))

    print(f'\nsaturated fraction (%), mean over {len(per_run)} seeds')
    print(f'{"":<10}' + ''.join(f'{t:>9g}' for t in SAT_GRID))
    for l in LAYERS:
        print(f'layer {l}   ' + ''.join(f'{v:>9.1f}' for v in sat_stats[l][0]))

    print('\nhighest per-neuron rate observed, by layer')
    for l in LAYERS:
        print(f'  layer {l}: ' + '  '.join(f'{run[l]["max_rate"]:.3f}'
                                           for run in per_run))

    # a factor of two either side of the adopted cutoff: no plateau, the dead
    # fraction moves by several points in every layer
    i = DEAD_GRID.index(DEAD_THRESHOLD)
    lo, hi = DEAD_GRID[i - 1], DEAD_GRID[i + 1]
    print(f'\nbetween {lo:g} and {hi:g} the dead fraction moves by, in points:')
    for l in LAYERS:
        mean = dead_stats[l][0]
        print(f'  layer {l}: {mean[i + 1] - mean[i - 1]:+.2f}')

    table = write_table(per_run, tags, TABLES / 'dead_neuron_sweep.csv')
    figures = draw(dead_stats, sat_stats, args.label, args.out_dir)
    print('\n' + '\n'.join(str(p) for p in [table, *figures]))


if __name__ == '__main__':
    main()