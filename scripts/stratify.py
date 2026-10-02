"""
Does the explanation degrade evenly, or only on the hard glitches?

THE QUESTION
------------
An overall divergence number averages over a population that is not uniform. A
model may keep its explanation on loud, obvious glitches and lose it on faint
ones, and the mean would show a moderate degradation everywhere instead — which
is a different claim, and the wrong one.

This stratifies the SAM divergence by the physical properties of the glitch:
signal-to-noise ratio, peak frequency, duration, bandwidth. Those come from the
Gravity Spy metadata already in selected_dataset.csv, so they cost nothing and
they describe the EVENT rather than the rendering of it. Image statistics would
describe the PNG; "the explanation degrades on low-SNR glitches" is a statement
a physicist can act on, "on dim images" is not.

WHY THE QUANTILES ARE WITHIN CLASS, AND WHY THAT IS NOT OPTIONAL
----------------------------------------------------------------
The four classes differ enormously in these quantities. Median SNR runs from
10.4 for Violin_Mode to 880.8 for Extremely_Loud, a factor of 85.

A global tertile split on SNR therefore does not split by intensity, it splits
by class: measured on this dataset, the top global tertile is 71.9 %
Extremely_Loud and the bottom is 1.1 %. Any "effect of SNR" found that way is
a class effect wearing a disguise.

Quantiles are taken WITHIN each class, which makes the cells balanced by
construction — about 117 validation samples per (class, tertile) cell — and
makes the comparison one of intensity at fixed class.

HOW THE METADATA JOINS TO THE IMAGES
------------------------------------
03_image_downloader.py names each file glitch_{index}.png using the GLOBAL
dataframe index, and saves it under its own class folder. So the integer in the
filename is the row number in selected_dataset.csv, and the folder name must
equal that row's ml_label.

That second condition is a real check rather than a formality: if the CSV were
ever re-sorted or regenerated, the indices would still parse and the join would
silently attach the wrong physics to every image. This script verifies it on
every run and refuses to continue if it fails.

ORDER
-----
The sample order is taken from the dataloader itself, not reconstructed. File
listings sort glitch_10.png before glitch_2.png, so any attempt to rebuild the
order by parsing names would be wrong in a way that produces a complete and
plausible table. The script also asserts the loader is not shuffled, since a
shuffled validation loader would make the row indices meaningless.

Validation set only.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import SequentialSampler

from scripts.dataloader import build_dataloaders
from xai.divergence_metrics import load_quantised_run, compare_stacks
from xai.sam import MANIFEST, load_reference_model
from xai.sam_metrics import sam_stacks

METADATA_CSV = Path('data/processed/selected_dataset.csv')
OUT_JSON = Path('results/stratified.json')
FEATURES_CSV = Path('results/validation_features.csv')
RUNS_DIR = Path('runs')

PHYSICAL = ['snr', 'peak_frequency', 'duration', 'bandwidth']
N_BINS = 3
BIN_NAMES = ['low', 'mid', 'high']
N_PER_CELL = 15


# --------------------------------------------------------------------------- #
# The feature table
# --------------------------------------------------------------------------- #
def load_metadata(path):
    """selected_dataset.csv, as a list of dicts indexed by row number."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in PHYSICAL:
            row[key] = float(row[key])
    return rows


def validation_paths(dataloader):
    """The file path of every validation sample, in loader order.

    Read from the dataset rather than rebuilt, for the reason in the docstring.
    """
    if not isinstance(dataloader.sampler, SequentialSampler):
        raise SystemExit(
            'the validation loader is shuffled, so sample positions are not '
            'stable between passes and nothing here can be joined to anything.')

    subset = dataloader.dataset
    if hasattr(subset, 'indices') and hasattr(subset, 'dataset'):
        base = subset.dataset
        return [base.samples[i][0] for i in subset.indices]
    return [p for p, _ in subset.samples]


def build_table(dataloader, metadata):
    """One row per validation sample: physical properties, joined and verified."""
    paths = validation_paths(dataloader)
    classes = (dataloader.dataset.dataset.classes
               if hasattr(dataloader.dataset, 'dataset')
               else dataloader.dataset.classes)

    table = []
    for position, path in enumerate(paths):
        parts = Path(path).parts
        folder, filename = parts[-2], parts[-1]
        match = re.fullmatch(r'glitch_(\d+)\.png', filename)
        if match is None:
            raise SystemExit(f'unexpected file name: {filename}')
        row_number = int(match.group(1))

        if row_number >= len(metadata):
            raise SystemExit(
                f'{filename} points at row {row_number}, past the end of the '
                f'metadata ({len(metadata)} rows). The CSV is not the one these '
                'images were downloaded from.')

        meta = metadata[row_number]
        if meta['ml_label'] != folder:
            raise SystemExit(
                f'join check failed on {path}: row {row_number} of the metadata '
                f'is {meta["ml_label"]}, the file is under {folder}.\n'
                'The indices still parse, so without this check every image '
                'would have been given the wrong physics. Do not continue.')

        table.append({
            'index': position,
            'path': path,
            'row': row_number,
            'class': folder,
            'label': classes.index(folder),
            'gravityspy_id': meta['gravityspy_id'],
            **{key: meta[key] for key in PHYSICAL},
        })
    return table


def assign_bins(table, column, n_bins=N_BINS):
    """Within-class quantile bins. See the docstring for why not global."""
    for name in {row['class'] for row in table}:
        rows = [r for r in table if r['class'] == name]
        values = np.array([r[column] for r in rows])
        edges = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
        for row in rows:
            row[f'{column}_bin'] = BIN_NAMES[int(np.searchsorted(edges, row[column],
                                                                side='right'))]
    return table


def stratified_sample(table, column, n_per_cell=N_PER_CELL, seed=0):
    """n_per_cell samples from each (class, bin), drawn reproducibly."""
    rng = np.random.default_rng(seed)
    chosen = {}
    for name in sorted({row['class'] for row in table}):
        for bin_name in BIN_NAMES:
            cell = [r for r in table
                    if r['class'] == name and r[f'{column}_bin'] == bin_name]
            if not cell:
                continue
            take = min(n_per_cell, len(cell))
            picked = rng.choice(len(cell), size=take, replace=False)
            chosen[(name, bin_name)] = [cell[i] for i in sorted(picked)]
    return chosen


# --------------------------------------------------------------------------- #
# Divergence, per cell
# --------------------------------------------------------------------------- #
def divergence_by_cell(reference, model, cells, dataset, time_steps, device,
                       layer_index, gamma, column):
    """SAM divergence between two models, per (class, bin) cell.

    Both models see the same images, cell by cell, so a difference between
    cells is a difference in the glitches rather than in the sampling.
    """
    results = {}
    for (class_name, bin_name), rows in cells.items():
        images = [dataset[row['index']][0] for row in rows]
        stacks_a = sam_stacks(reference, images, time_steps, device,
                              layer_index, gamma)
        stacks_b = sam_stacks(model, images, time_steps, device,
                              layer_index, gamma)
        per_image = [compare_stacks(a, b, use_ssim=False)
                     for a, b in zip(stacks_a, stacks_b)]

        com_shift = [m['com_b'] - m['com_a'] for m in per_image]
        results[f'{class_name}|{bin_name}'] = {
            'class': class_name,
            'bin': bin_name,
            'n': len(per_image),
            'pcc': float(np.nanmean([m['pcc'] for m in per_image])),
            'iou': float(np.nanmean([m['iou'] for m in per_image])),
            'com_shift': float(np.nanmean(com_shift)),
            # What 'low' and 'high' mean numerically in this cell. Needed in the
            # report: the same bin name covers very different physical ranges
            # from one class to the next.
            f'median_{column}': float(np.median([row[column] for row in rows])),
        }
    return results


# --------------------------------------------------------------------------- #
def write_features(table, column):
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURES_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    print(f'feature table written to {FEATURES_CSV}')


def report_design(table, column):
    """What the stratification looks like, before any model is involved."""
    classes = sorted({row['class'] for row in table})
    print(f'\nstratifying on {column}, within-class {N_BINS}-quantiles')
    print(f'{"class":<18}' + ''.join(f'{b:>10}' for b in BIN_NAMES)
          + f'{"median " + column:>18}')
    for name in classes:
        rows = [r for r in table if r['class'] == name]
        counts = [sum(1 for r in rows if r[f'{column}_bin'] == b) for b in BIN_NAMES]
        medians = [np.median([r[column] for r in rows if r[f'{column}_bin'] == b])
                   for b in BIN_NAMES]
        print(f'{name:<18}' + ''.join(f'{c:>10}' for c in counts)
              + '   ' + ' '.join(f'{m:>8.1f}' for m in medians))
    print('  cells are balanced by construction; the medians show the range each')
    print('  class actually spans, which differs enormously between classes.')


def main():
    parser = argparse.ArgumentParser(
        description='SAM divergence stratified by glitch physics.')
    parser.add_argument('--column', choices=PHYSICAL, default='snr')
    parser.add_argument('--metadata', type=Path, default=METADATA_CSV)
    parser.add_argument('--n-per-cell', type=int, default=N_PER_CELL)
    parser.add_argument('--features-only', action='store_true',
                        help='build and check the feature table, then stop. '
                             'Needs no model and no GPU.')
    parser.add_argument('--runs', type=str, nargs='+', default=None,
                        help='run directory names; default: every qat_* run')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reference, time_steps, input_size, layer_index, gamma = load_reference_model(
        device, MANIFEST)
    _, _, val_dataloader, _ = build_dataloaders(input_size=input_size, verbose=False)

    metadata = load_metadata(args.metadata)
    table = build_table(val_dataloader, metadata)
    print(f'{len(table)} validation samples joined to {len(metadata)} metadata '
          'rows, class check passed')

    table = assign_bins(table, args.column)
    write_features(table, args.column)
    report_design(table, args.column)

    if args.features_only:
        return

    cells = stratified_sample(table, args.column, args.n_per_cell)
    dataset = val_dataloader.dataset

    run_dirs = ([RUNS_DIR / name for name in args.runs] if args.runs else
                sorted(d for d in RUNS_DIR.glob('qat_*') if (d / 'run.json').exists()))
    if not run_dirs:
        print('\nno qat_* runs yet — the feature table and its design are written,')
        print('and the divergence can be added later without recomputing them.')
        return

    output = {'column': args.column, 'n_bins': N_BINS,
              'n_per_cell': args.n_per_cell, 'runs': {}}

    for run_dir in run_dirs:
        net, handle, summary = load_quantised_run(run_dir, device)
        results = divergence_by_cell(reference, net, cells, dataset, time_steps,
                                     device, layer_index, gamma, args.column)
        handle.remove()
        output['runs'][run_dir.name] = {'arm': summary['arm'], **{'cells': results}}

        print(f'\n{run_dir.name}  ({summary["arm"]})')
        print(f'{"class":<18}' + ''.join(f'{b + " PCC":>12}' for b in BIN_NAMES)
              + f'{"low-high":>11}')
        for name in sorted({c['class'] for c in results.values()}):
            values = [results[f'{name}|{b}']['pcc'] for b in BIN_NAMES]
            print(f'{name:<18}' + ''.join(f'{v:>12.3f}' for v in values)
                  + f'{values[0] - values[-1]:>+11.3f}')

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print('\nThe last column is the low-minus-high difference. Positive means the')
    print('explanation survives BETTER on faint glitches than on loud ones, which')
    print('would be surprising; negative means it degrades on the faint ones,')
    print('which is the prediction worth stating before reading the numbers.')
    print(f'\nwritten to {OUT_JSON}')


if __name__ == '__main__':
    main()