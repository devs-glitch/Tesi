'''
Figures of the reference network's internal dynamics
'''

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from scripts.dataloader import build_dataloaders
from scripts.figure_style import (BLUE, GREEN, INK, ORANGE,
                                  FIG_DIR, clean_axes, frame, save,
                                  use_thesis_style)
from scripts.training_utils import direct_encode
from xai.sam import load_reference_model

import matplotlib.pyplot as plt

MANIFEST = 'config/baseline_manifest.json'

# Same constants as scripts/evaluate_model.py; the two must not drift apart
DEAD_THRESHOLD = 0.01
SATURATED_THRESHOLD = 0.80
N_NEURONS_SAMPLE = 200
MIN_INTERVALS = 5
CV_SEED = 42

LAYERS = (1, 2, 3, 4)

use_thesis_style()


# Measurement

def collect_statistics(net, dataloader, time_steps, device, layers=LAYERS):
    # Predictions, per-neuron firing rates and pooled ISI in one pass
    random.seed(CV_SEED)
    net.eval()

    predictions, truths = [], []
    rate_sums = {l: None for l in layers}
    positions = {l: None for l in layers}
    isi_pools = {l: None for l in layers}
    n_seen = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            spike_out, *hidden = net(direct_encode(images, time_steps))
            spikes_of = dict(zip(LAYERS, hidden))

            predictions.extend(
                torch.argmax(spike_out.sum(dim=0), dim=1).cpu().numpy())
            truths.extend(labels.cpu().numpy())

            batch = images.shape[0]
            n_seen += batch

            for layer in layers:
                spikes = spikes_of[layer]            # [T, B, C, H, W]

                per_neuron = spikes.mean(dim=(0, 1)) * batch
                rate_sums[layer] = (per_neuron if rate_sums[layer] is None
                                    else rate_sums[layer] + per_neuron)

                # The sampled positions are fixed once, on the first batch, so
                # every image contributes intervals of the same neurons
                if positions[layer] is None:
                    channels, height, width = spikes.shape[2:]
                    grid = [(c, h, w) for c in range(channels)
                            for h in range(height) for w in range(width)]
                    positions[layer] = random.sample(
                        grid, min(N_NEURONS_SAMPLE, len(grid)))
                    isi_pools[layer] = {p: [] for p in positions[layer]}

                index = torch.tensor(positions[layer], device=spikes.device)
                trains = spikes[:, :, index[:, 0], index[:, 1],
                                index[:, 2]].cpu().numpy()   # [T, B, N]

                for b in range(batch):
                    for n, position in enumerate(positions[layer]):
                        times = np.flatnonzero(trains[:, b, n])
                        if times.size >= 2:
                            isi_pools[layer][position].extend(
                                np.diff(times).tolist())

    rates = {l: (rate_sums[l] / n_seen).cpu().numpy() for l in layers}
    return np.asarray(truths), np.asarray(predictions), rates, isi_pools


def cv_per_position(isi_pools, layer):
    # CV of each sampled position, where enough intervals were observed
    values = []
    for intervals in isi_pools[layer].values():
        if len(intervals) < MIN_INTERVALS:
            continue
        arr = np.asarray(intervals, dtype=float)
        if arr.mean() > 0:
            values.append(float(arr.std() / arr.mean()))
    return np.asarray(values)


# Figures

def plot_confusion_matrix(truths, predictions, class_names, out_dir=FIG_DIR):
    counts = confusion_matrix(truths, predictions)
    support = counts.sum(axis=1)
    shares = counts / support[:, None]

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    image = ax.imshow(shares, cmap='Blues', vmin=0, vmax=1)

    # Three decimals
    for i in range(shares.shape[0]):
        for j in range(shares.shape[1]):
            ax.text(j, i, f'{shares[i, j]:.3f}', ha='center', va='center',
                    fontsize=9, color='white' if shares[i, j] > 0.5 else INK)

    names = [c.replace('_', ' ') for c in class_names]
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8.5)
    ax.set_yticklabels([f'{n}\n$n = {s}$' for n, s in zip(names, support)],
                       fontsize=8.5)
    ax.set_xlabel('predicted label', fontsize=9, labelpad=4)
    ax.set_ylabel('true label', fontsize=9, labelpad=4)
    ax.tick_params(length=0)
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)

    bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8)

    frame(fig, 'Confusion matrix, normalized per true class', pad=0.45)
    save(fig, 'confusion_matrix', out_dir)


def plot_firing_rates(rates, out_dir=FIG_DIR):
    layers = sorted(rates)
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.6), sharex=True, sharey=True)
    axes = axes.flat

    for ax, layer in zip(axes, layers):
        values = rates[layer].flatten()
        n_dead = int((values < DEAD_THRESHOLD).sum())
        n_saturated = int((values > SATURATED_THRESHOLD).sum())

        ax.hist(values, bins=50, range=(0, 1), color=BLUE, edgecolor='white',
                linewidth=0.3, zorder=3)
        for threshold in (DEAD_THRESHOLD, SATURATED_THRESHOLD):
            ax.axvline(threshold, color=ORANGE, linestyle=(0, (4, 2)),
                       linewidth=1.2, zorder=4)

        # A log count axis
        ax.set_yscale('log')
        ax.set_xlim(0, 1)
        ax.set_ylim(0.7, ax.get_ylim()[1] * 1.6)
        clean_axes(ax)

        # Counts beside the percentages
        ax.set_title(
            f'Layer {layer} — {values.size:,} neurons\n'
            f'dead {n_dead:,} ({100 * n_dead / values.size:.1f}%)   '
            f'saturated {n_saturated:,} '
            f'({100 * n_saturated / values.size:.2f}%)',
            fontsize=8.4, pad=7, linespacing=1.5)

    axes[0].set_ylabel('neurons (log scale)', fontsize=8.5)
    fig.supxlabel('firing rate per neuron', fontsize=8.5, y=0.015)

    frame(fig, 'Firing rate per neuron, by layer',
          note=f'Reference checkpoint on validation. Dashed lines: the dead '
               f'({DEAD_THRESHOLD}) and saturated ({SATURATED_THRESHOLD}) '
               f'thresholds.')
    save(fig, 'firing_rate_distributions', out_dir)


def plot_cv_isi(isi_pools, out_dir=FIG_DIR):
    layers = sorted(isi_pools)
    per_layer = {l: cv_per_position(isi_pools, l) for l in layers}

    pooled = np.concatenate([v for v in per_layer.values() if v.size])
    if pooled.size:
        span = max(float(pooled.max() - pooled.min()), 1e-6)
        low, high = float(pooled.min()) - 0.06 * span, float(pooled.max()) + 0.06 * span
    else:
        low, high = 0.0, 1.0

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.6), sharex=True, sharey=True)
    axes = axes.flat

    for ax, layer in zip(axes, layers):
        values = per_layer[layer]
        if values.size:
            # One range shared by the four panels
            ax.hist(values, bins=28, range=(low, high), color=GREEN,
                    edgecolor='white', linewidth=0.3, zorder=3)
            median = float(np.median(values))
            ax.axvline(median, color=INK, linestyle=(0, (4, 2)), linewidth=1.2,
                       zorder=4)
            ax.set_title(f'Layer {layer} — {values.size} positions\n'
                         f'median {median:.2f}', fontsize=8.4, pad=7,
                         linespacing=1.5)
        else:
            ax.set_title(f'Layer {layer} — too few intervals\n ',
                         fontsize=8.4, pad=7, linespacing=1.5)
        ax.set_xlim(low, high)
        clean_axes(ax)

    axes[0].set_ylabel('positions', fontsize=8.5)
    fig.supxlabel('CV of the inter-spike interval', fontsize=8.5, y=0.015)

    frame(fig, 'Regularity of firing, by layer',
          note=f'{N_NEURONS_SAMPLE} sampled positions per layer, intervals '
               f'pooled over the validation partition, CV read where at least '
               f'{MIN_INTERVALS} intervals were observed. Dashed line: the '
               f'median.')
    save(fig, 'cv_isi_distributions', out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=FIG_DIR)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, time_steps, input_size, layer_index, gamma = load_reference_model(
        device, MANIFEST)
    dataset, _, val_dataloader, _ = build_dataloaders(input_size=input_size,
                                                      verbose=False)
    class_names = dataset.classes
    print(f'{input_size} px | layer {layer_index} | gamma {gamma} | '
          f'{len(val_dataloader.dataset)} validation images')

    truths, predictions, rates, isi_pools = collect_statistics(
        net, val_dataloader, time_steps, device)

    print(f'accuracy {float((truths == predictions).mean()):.4f}')
    for layer in LAYERS:
        values = rates[layer].flatten()
        cvs = cv_per_position(isi_pools, layer)
        print(f'  layer {layer}: mean rate {values.mean():.4f} | '
              f'dead {100 * (values < DEAD_THRESHOLD).mean():.1f}% | '
              f'saturated {int((values > SATURATED_THRESHOLD).sum())} neurons | '
              f'CV {np.median(cvs):.3f} on {cvs.size} positions')

    plot_confusion_matrix(truths, predictions, class_names, args.out_dir)
    plot_firing_rates(rates, args.out_dir)
    plot_cv_isi(isi_pools, args.out_dir)


if __name__ == '__main__':
    main()