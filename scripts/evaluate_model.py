'''
One validation pass per model collecting everything that pass can give
'''

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.dataloader import build_dataloaders
from scripts.training_utils import direct_encode
from xai.divergence_metrics import load_quantised_run
from xai.sam import MANIFEST, load_reference_model

OUT_DIR = Path('results/grid')
RUNS_DIR = Path('runs')

DEAD_THRESHOLD = 0.01       
SATURATED_THRESHOLD = 0.90  


def cv_isi_from_train(spike_train):
    # CV of inter-spike intervals for one neuron over one sample. spike_train: 1-D binary tensor of length T

    times = np.flatnonzero(spike_train)
    if times.size < 3:
        return float('nan')
    intervals = np.diff(times)
    mean = intervals.mean()
    if mean == 0:
        return float('nan')
    return float(intervals.std() / mean)


def evaluate(net, dataloader, time_steps, device, cv_sample_cap=4096):
    # returns aggregates, per-neuron rate vectors, per-sample rows

    net.eval()
    correct = total = 0
    row_index = 0

    rate_sums = {}          # per-neuron spike-rate accumulator, per layer
    weighted_time = {}
    spike_total = {}
    cv_values = {}
    cv_attempted = {}
    rows = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            spike_out, *hidden = net(direct_encode(images, time_steps))

            predictions = spike_out.sum(dim=0).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()

            batch = images.shape[0]
            per_sample_rate = {}
            per_sample_com = {}

            for index, spikes in enumerate(hidden, start=1):
                # spikes: [T, B, C, H, W]
                if index not in rate_sums:
                    rate_sums[index] = torch.zeros(
                        spikes.shape[2:].numel(), dtype=torch.float64)
                    weighted_time[index] = 0.0
                    spike_total[index] = 0.0
                    cv_values[index] = []
                    cv_attempted[index] = 0

                # per neuron, averaged over time, summed over the batch
                rate_sums[index] += (spikes.mean(dim=0)
                                     .reshape(batch, -1).sum(dim=0)
                                     .double().cpu())

                steps = torch.arange(spikes.shape[0], dtype=spikes.dtype,
                                     device=spikes.device).view(-1, 1, 1, 1, 1)
                weighted_time[index] += float((steps * spikes).sum())
                spike_total[index] += float(spikes.sum())

                # per sample, for the CSV
                flat = spikes.reshape(spikes.shape[0], batch, -1)   # [T, B, N]
                per_sample_rate[index] = flat.mean(dim=(0, 2)).cpu().numpy()
                energy = flat.sum(dim=2)                            # [T, B]
                totals = energy.sum(dim=0)
                step_axis = torch.arange(spikes.shape[0], dtype=energy.dtype,
                                         device=energy.device).view(-1, 1)
                com = torch.where(totals > 0, (step_axis * energy).sum(dim=0) / totals,
                                  torch.full_like(totals, float('nan')))
                per_sample_com[index] = com.cpu().numpy()

                # CV-ISI on a fixed subset of neurons, first sample of the batch
                subset = flat[:, 0, :cv_sample_cap].cpu().numpy()
                for neuron in range(subset.shape[1]):
                    cv_attempted[index] += 1
                    value = cv_isi_from_train(subset[:, neuron])
                    if not np.isnan(value):
                        cv_values[index].append(value)

            for b in range(batch):
                row = {
                    'index': row_index,
                    'label': int(labels[b].item()),
                    'prediction': int(predictions[b].item()),
                    'correct': int(predictions[b].item() == labels[b].item()),
                }
                for index in per_sample_rate:
                    row[f'rate_layer{index}'] = float(per_sample_rate[index][b])
                    row[f'com_layer{index}'] = float(per_sample_com[index][b])
                rows.append(row)
                row_index += 1

    n_seen = total
    rates = {i: (v / n_seen).numpy() for i, v in rate_sums.items()}

    aggregates = {
        'accuracy': correct / total,
        'n_val': total,
        'mean_rate': {i: float(r.mean()) for i, r in rates.items()},
        'dead_fraction': {i: float((r < DEAD_THRESHOLD).mean())
                          for i, r in rates.items()},
        'saturated_fraction': {i: float((r > SATURATED_THRESHOLD).mean())
                               for i, r in rates.items()},
        'n_neurons': {i: int(r.size) for i, r in rates.items()},
        'spike_com': {i: (weighted_time[i] / spike_total[i]
                          if spike_total[i] > 0 else float('nan'))
                      for i in weighted_time},
        'cv_isi': {i: (float(np.mean(v)) if v else float('nan'))
                   for i, v in cv_values.items()},
        'cv_isi_defined_fraction': {i: (len(cv_values[i]) / cv_attempted[i]
                                        if cv_attempted[i] else 0.0)
                                    for i in cv_values},
    }
    return aggregates, rates, rows

def load_model(name, device):

    if name in ('fp32', 'reference'):
        net, time_steps, input_size, _, _ = load_reference_model(device, MANIFEST)
        return net, None, time_steps, input_size, {'arm': 'fp32-reference'}

    run_dir = RUNS_DIR / name
    if not (run_dir / 'run.json').exists():
        raise FileNotFoundError(f'no run.json in {run_dir}')
    net, handle, summary = load_quantised_run(run_dir, device)
    return net, handle, summary['hyperparameters']['time_steps'], \
        summary['input_size'], summary


def write_outputs(tag, aggregates, rates, rows, metadata):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / f'{tag}.json', 'w', encoding='utf-8') as f:
        json.dump({**metadata, **aggregates}, f, indent=2)

    np.savez_compressed(OUT_DIR / f'{tag}_rates.npz',
                        **{f'layer{i}': r for i, r in rates.items()})

    with open(OUT_DIR / f'{tag}_per_sample.csv', 'w', newline='',
              encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def report(tag, aggregates):
    layers = sorted(aggregates['mean_rate'])
    print(f'\n{tag}  accuracy {aggregates["accuracy"]:.4f}')
    print(f'{"":<22}' + ''.join(f'{"layer" + str(i):>12}' for i in layers))
    for label, key, fmt in (
            ('mean firing rate', 'mean_rate', '{:>12.4f}'),
            ('dead fraction', 'dead_fraction', '{:>11.1%} '),
            ('saturated fraction', 'saturated_fraction', '{:>11.1%} '),
            ('spike CoM (steps)', 'spike_com', '{:>12.3f}'),
            ('CV-ISI', 'cv_isi', '{:>12.3f}'),
            ('  where defined', 'cv_isi_defined_fraction', '{:>11.2%} ')):
        print(f'{label:<22}' + ''.join(fmt.format(aggregates[key][i]) for i in layers))


def main():
    parser = argparse.ArgumentParser(
        description='One validation pass per model')
    parser.add_argument('models', nargs='*', default=None,
                        help='run directory names, or "fp32" for the manifest '
                             'reference. Default: fp32 plus every qat_* run.')
    parser.add_argument('--force', action='store_true',
                        help='recompute models whose output files already exist')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    names = args.models or (
        ['fp32'] + sorted(d.name for d in RUNS_DIR.glob('qat_*')
                          if (d / 'run.json').exists()))

    for name in names:
        tag = name
        if not args.force and (OUT_DIR / f'{tag}.json').exists():
            print(f'[{tag}] already present: skipping')
            continue

        net, handle, time_steps, input_size, summary = load_model(name, device)
        _, _, val_dataloader, _ = build_dataloaders(input_size=input_size,
                                                    verbose=False)
        aggregates, rates, rows = evaluate(net, val_dataloader, time_steps, device)
        if handle is not None:
            handle.remove()

        assert len(rows) == len(val_dataloader.dataset), (
            f'{len(rows)} rows for {len(val_dataloader.dataset)} images - the '
            'per-sample files will not join across models')

        write_outputs(tag, aggregates, rates, rows, {
            'tag': tag,
            'arm': summary.get('arm'),
            'weight_bits': summary.get('quantisation', {}).get('weight_bits'),
            'membrane_bits': summary.get('quantisation', {}).get('membrane_bits'),
            'seed': summary.get('seed'),
            'input_size': input_size,
            'time_steps': time_steps,
            'dead_threshold': DEAD_THRESHOLD,
            'saturated_threshold': SATURATED_THRESHOLD,
        })
        report(tag, aggregates)

    print(f'\nwritten to {OUT_DIR}/')


if __name__ == '__main__':
    main()