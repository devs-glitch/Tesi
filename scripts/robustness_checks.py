'''
Robustness checks on the quantisation grid, run after every other analysis

  ties        how many predictions of each model come from a tie in the output
              spike counts (argmax returns the first class, Blip, on a tie)
  cv          per-image CV-ISI of the controls that per_image_cv.py did not
              cover: membrane at 4 bits and weights at 2 bits
  iou         95 % bootstrap intervals of the per-cell IoU, PCC and |CoM shift|
              of every quantised model against its own parent
  deletion    95 % bootstrap intervals of the deletion gaps of the 2-bit
              membrane models (parent baseline, own map, parent's map)

Every check reuses the functions that produced the published numbers, with the
same images and the same random generators, and verifies that its means
reproduce the stored results before reporting intervals.

Run from the repository root, with the checkpoints in runs/:
    python -m scripts.robustness_checks all
    python -m scripts.robustness_checks ties          (one check at a time)
Outputs: results/robustness/<check>.json, and a summary on the console.
'''

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.dataloader import build_dataloaders
from scripts.evaluate_model import load_model
from scripts.per_image_cv import pattern_histogram, pattern_table, summarise
from scripts.training_utils import direct_encode
from xai.cross_faithfulness import (compute_maps, deletion_curve, auc,
                                    predicted_class)
from xai.divergence_metrics import (compare_stacks, load_fp32_for_seed,
                                    load_quantised_run)
from xai.sam import MANIFEST, load_reference_model
from xai.sam_metrics import samples_per_class, sam_stacks

RUNS_DIR = Path('runs')
RESULTS = Path('results')
OUT_DIR = RESULTS / 'robustness'

SEEDS = (1234, 2345, 3456)
ARMS = ('w8ux', 'wxu8', 'w8u8', 'w4ux', 'wxu4', 'w4u4', 'w2ux', 'wxu2', 'w2u2')

N_BOOT = 2000
BOOT_SEED = 0
TOLERANCE = 1e-5          # stored means must be reproduced to this precision

CV_CONTROLS = tuple(f'qat_{arm}_112px_seed{s}'
                    for arm in ('wxu4', 'w2ux') for s in SEEDS)


# helpers

def write(name, payload):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f'{name}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f'\nwritten to {path}')


def bootstrap_mean(values, rng, n_boot=N_BOOT):
    # percentile interval of the mean, resampling the images of one cell
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {'mean': float('nan'), 'low': float('nan'), 'high': float('nan'), 'n': 0}
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {'mean': float(values.mean()), 'low': float(low), 'high': float(high),
            'n': int(values.size)}


def bootstrap_mean_of_cells(cells, rng, n_boot=N_BOOT):
    # interval of the mean over cells, each cell resampled independently;
    # seeds and classes are treated as fixed, images as the sampled unit
    cells = [np.asarray(c, dtype=float)[~np.isnan(np.asarray(c, dtype=float))]
             for c in cells]
    cells = [c for c in cells if c.size]
    draws = np.mean([rng.choice(c, size=(n_boot, c.size), replace=True).mean(axis=1)
                     for c in cells], axis=0)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {'mean': float(np.mean([c.mean() for c in cells])),
            'low': float(low), 'high': float(high), 'n_cells': len(cells)}


def check_reproduced(label, measured, stored):
    gap = abs(measured - stored)
    status = 'ok' if gap <= TOLERANCE else 'MISMATCH'
    if status != 'ok':
        print(f'  {status}: {label} measured {measured:.6f}, stored {stored:.6f}')
    return gap


def validation(input_size):
    dataset, _, val_dataloader, _ = build_dataloaders(input_size=input_size,
                                                      verbose=False)
    return dataset, val_dataloader


# ties in the readout

def check_ties(device, input_size):
    _, val_dataloader = validation(input_size)
    names = [f'fp32_{input_size}px_seed{s}' for s in SEEDS] + \
            [f'qat_{a}_{input_size}px_seed{s}' for a in ARMS for s in SEEDS]

    output = {}
    print(f'\n{"model":28s}{"acc":>8}{"ties":>7}{"silent":>8}{"pred 0":>8}'
          f'{"0 from ties":>13}')
    for name in names:
        if not (RUNS_DIR / name / 'run.json').exists():
            print(f'{name:28s} missing, skipped')
            continue
        net, handle, time_steps, _, _ = load_model(name, device)
        n = correct = ties = silent = 0
        pred_counts = np.zeros(4, dtype=int)
        tie_pred_counts = np.zeros(4, dtype=int)
        correct_untied = untied = 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                spike_out, *_ = net(direct_encode(images, time_steps))
                counts = spike_out.sum(dim=0)                      # [B, classes]
                top = counts.max(dim=1).values
                n_top = (counts == top.unsqueeze(1)).sum(dim=1)
                tied = n_top > 1
                predictions = counts.argmax(dim=1)

                n += labels.numel()
                correct += int((predictions == labels).sum())
                ties += int(tied.sum())
                silent += int((top == 0).sum())
                untied += int((~tied).sum())
                correct_untied += int(((predictions == labels) & ~tied).sum())
                for c in range(counts.shape[1]):
                    pred_counts[c] += int((predictions == c).sum())
                    tie_pred_counts[c] += int(((predictions == c) & tied).sum())
        if handle is not None:
            handle.remove()

        output[name] = {
            'n': n, 'accuracy': correct / n,
            'tied': ties, 'tied_fraction': ties / n,
            'silent_output': silent,
            'predictions_per_class': pred_counts.tolist(),
            'tied_predictions_per_class': tie_pred_counts.tolist(),
            'accuracy_untied': correct_untied / untied if untied else float('nan'),
        }
        print(f'{name:28s}{correct / n:>8.4f}{ties:>7d}{silent:>8d}'
              f'{pred_counts[0]:>8d}{tie_pred_counts[0]:>13d}')
    write('ties', output)
    return output


# per-image CV of the controls

def check_cv(device, input_size):
    _, val_dataloader = validation(input_size)
    stored = json.loads((RESULTS / 'per_image_cv.json').read_text(encoding='utf-8'))
    output = {}
    tables = {}
    for name in CV_CONTROLS:
        if not (RUNS_DIR / name / 'run.json').exists():
            print(f'{name:28s} missing, skipped')
            continue
        net, handle, time_steps, _, summary = load_model(name, device)
        if time_steps not in tables:
            tables[time_steps] = pattern_table(time_steps)
        n_spikes, cv = tables[time_steps]
        counts = pattern_histogram(net, val_dataloader, time_steps, device)
        if handle is not None:
            handle.remove()
        output[name] = {'arm': summary.get('arm'), 'seed': summary.get('seed'),
                        'time_steps': time_steps,
                        'layers': summarise(counts, n_spikes, cv)}

    print(f'\n{"layer 1":28s}{">=3 spikes":>12}{"exactly 0":>11}{"mean CV":>9}')
    for name in list(stored) + list(output):
        layer = (output.get(name) or stored[name])['layers']['1']
        print(f'{name:28s}{layer["fraction_with_two_intervals"]:>12.3f}'
              f'{layer["fraction_cv_exactly_zero"]:>11.3f}'
              f'{layer["mean_cv_from_two_intervals"]:>9.3f}')
    write('per_image_cv_controls', output)
    return output


# bootstrap of the divergence metrics

def check_iou(device, input_size, n_per_class=20):
    reference, time_steps, _, layer_index, gamma = load_reference_model(device, MANIFEST)
    dataset, val_dataloader = validation(input_size)
    classes = dataset.classes
    samples = samples_per_class(val_dataloader, len(classes), n_per_class)
    stored = json.loads((RESULTS / 'divergence.json').read_text(encoding='utf-8'))['runs']
    rng = np.random.default_rng(BOOT_SEED)

    parent_stacks = {}
    per_image = {}
    cells = {}
    worst_gap = 0.0
    for seed in SEEDS:
        parent = load_fp32_for_seed(seed, input_size, device)
        parent_stacks[seed] = {c: sam_stacks(parent, samples[c], time_steps, device,
                                             layer_index, gamma)
                               for c in samples}
        for arm in ARMS:
            name = f'qat_{arm}_{input_size}px_seed{seed}'
            if not (RUNS_DIR / name / 'run.json').exists():
                print(f'{name} missing, skipped')
                continue
            net, handle, _ = load_quantised_run(RUNS_DIR / name, device)
            per_image[name] = {}
            cells[name] = {}
            for c, images in samples.items():
                stacks = sam_stacks(net, images, time_steps, device, layer_index, gamma)
                rows = [compare_stacks(a, b, use_ssim=False)
                        for a, b in zip(parent_stacks[seed][c], stacks)]
                iou = [r['iou'] for r in rows]
                pcc = [r['pcc'] for r in rows]
                com = [abs(r['com_b'] - r['com_a']) for r in rows]
                per_image[name][classes[c]] = {'iou': iou, 'pcc': pcc, 'com_abs': com}
                cells[name][classes[c]] = {
                    'iou': bootstrap_mean(iou, rng),
                    'pcc': bootstrap_mean(pcc, rng),
                    'com_abs': bootstrap_mean(com, rng),
                }
                if name in stored:
                    s = stored[name]['divergence'][classes[c]]
                    worst_gap = max(worst_gap,
                                    check_reproduced(f'{name} {classes[c]} IoU',
                                                     float(np.nanmean(iou)), s['iou']),
                                    check_reproduced(f'{name} {classes[c]} PCC',
                                                     float(np.nanmean(pcc)), s['pcc']))
            handle.remove()

    # mean over the twelve cells of each configuration
    summary = {}
    for arm in ARMS:
        names = [f'qat_{arm}_{input_size}px_seed{s}' for s in SEEDS]
        names = [n for n in names if n in per_image]
        if not names:
            continue
        summary[arm] = {metric: bootstrap_mean_of_cells(
            [per_image[n][c][metric] for n in names for c in per_image[n]], rng)
            for metric in ('iou', 'pcc', 'com_abs')}

    print(f'\nlargest gap from divergence.json: {worst_gap:.2e}')
    print(f'\n{"config":8s}{"IoU":>8}{"95% interval":>18}{"PCC":>9}{"95% interval":>20}')
    for arm, s in summary.items():
        i, p = s['iou'], s['pcc']
        print(f'{arm:8s}{i["mean"]:>8.3f}   [{i["low"]:.3f}, {i["high"]:.3f}]'
              f'{p["mean"]:>9.4f}   [{p["low"]:.4f}, {p["high"]:.4f}]')
    print('\n2-bit membrane, per cell (IoU, 95% interval):')
    for seed in SEEDS:
        name = f'qat_wxu2_{input_size}px_seed{seed}'
        for c, cell in cells.get(name, {}).items():
            i = cell['iou']
            print(f'  {seed} {c:18s}{i["mean"]:.3f}  [{i["low"]:.3f}, {i["high"]:.3f}]')

    write('bootstrap_iou', {'n_boot': N_BOOT, 'boot_seed': BOOT_SEED,
                            'n_per_class': n_per_class,
                            'reproduction_gap': worst_gap,
                            'per_configuration': summary, 'per_cell': cells,
                            'per_image': per_image})
    return summary


# bootstrap of the deletion gaps

def deletion_per_image(net, samples, maps, time_steps, device, targets, seed=0):
    # xai.cross_faithfulness.evaluate_model, keeping every image's AUCs;
    # same generator, same call order, hence the same random permutations
    generator = torch.Generator().manual_seed(seed)
    out = {}
    for label, images in samples.items():
        sam_aucs, random_aucs = [], []
        for index, image in enumerate(images):
            target = targets[label][index]
            order = torch.argsort(maps[label][index].flatten(), descending=True)
            sam_aucs.append(auc(deletion_curve(net, image, order, time_steps,
                                               device, target)))
            shuffled = torch.randperm(order.numel(), generator=generator)
            random_aucs.append(auc(deletion_curve(net, image, shuffled, time_steps,
                                                  device, target)))
        out[label] = {'auc_sam': sam_aucs, 'auc_random': random_aucs,
                      'delta': (np.array(sam_aucs) - np.array(random_aucs)).tolist()}
    return out


def check_deletion(device, input_size, n_per_class=15):
    _, time_steps, _, layer_index, gamma = load_reference_model(device, MANIFEST)
    dataset, val_dataloader = validation(input_size)
    classes = dataset.classes
    samples = samples_per_class(val_dataloader, len(classes), n_per_class)
    stored = json.loads((RESULTS / 'cross_faithfulness.json')
                        .read_text(encoding='utf-8'))['runs']
    rng = np.random.default_rng(BOOT_SEED)

    output = {}
    worst_gap = 0.0
    print(f'\n{"seed class":28s}{"baseline":>26}{"own map":>26}{"parent map":>26}'
          f'{"own - parent":>26}')
    for seed in SEEDS:
        name = f'qat_wxu2_{input_size}px_seed{seed}'
        if not (RUNS_DIR / name / 'run.json').exists():
            print(f'{name} missing, skipped')
            continue
        net, handle, _ = load_quantised_run(RUNS_DIR / name, device)
        parent = load_fp32_for_seed(seed, input_size, device)

        parent_maps = compute_maps(parent, samples, time_steps, device,
                                   layer_index, gamma, input_size)
        targets = {label: [predicted_class(parent, img, time_steps, device)
                           for img in images]
                   for label, images in samples.items()}
        baseline = deletion_per_image(parent, samples, parent_maps, time_steps,
                                      device, targets)
        own_maps = compute_maps(net, samples, time_steps, device, layer_index,
                                gamma, input_size)
        own = deletion_per_image(net, samples, own_maps, time_steps, device, targets)
        borrowed = deletion_per_image(net, samples, parent_maps, time_steps,
                                      device, targets)
        handle.remove()

        output[name] = {}
        for label in samples:
            c = classes[label]
            # the gap is reported as random minus SAM: larger is more faithful
            gap = {k: -np.array(v[label]['delta'])
                   for k, v in (('baseline', baseline), ('own', own),
                                ('parent_map', borrowed))}
            cell = {k: bootstrap_mean(v, rng) for k, v in gap.items()}
            cell['own_minus_parent_map'] = bootstrap_mean(gap['own'] - gap['parent_map'], rng)
            cell['per_image_gap'] = {k: v.tolist() for k, v in gap.items()}
            output[name][c] = cell

            if name in stored:
                s = stored[name]
                for key, stored_key in (('baseline', 'baseline'), ('own', 'own_maps'),
                                        ('parent_map', 'reference_maps')):
                    worst_gap = max(worst_gap, check_reproduced(
                        f'{name} {c} {key}', cell[key]['mean'],
                        -s[stored_key][c]['delta']))

            def fmt(x):
                return f'{x["mean"]:+.3f} [{x["low"]:+.3f},{x["high"]:+.3f}]'
            print(f'{seed} {c:23s}{fmt(cell["baseline"]):>26}{fmt(cell["own"]):>26}'
                  f'{fmt(cell["parent_map"]):>26}{fmt(cell["own_minus_parent_map"]):>26}')

    print(f'\nlargest gap from cross_faithfulness.json: {worst_gap:.2e}')
    write('bootstrap_deletion', {'n_boot': N_BOOT, 'boot_seed': BOOT_SEED,
                                 'n_per_class': n_per_class,
                                 'sign': 'gap = AUC_random - AUC_SAM, larger is more faithful',
                                 'reproduction_gap': worst_gap, 'runs': output})
    return output


CHECKS = {'ties': check_ties, 'cv': check_cv, 'iou': check_iou,
          'deletion': check_deletion}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('check', choices=list(CHECKS) + ['all'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest = json.loads(Path(MANIFEST).read_text(encoding='utf-8'))
    input_size = manifest['input_size']
    print(f'device {device} | {input_size} px')

    for name in (CHECKS if args.check == 'all' else [args.check]):
        print(f'\n{"=" * 30} {name} {"=" * 30}')
        CHECKS[name](device, input_size)


if __name__ == '__main__':
    main()
