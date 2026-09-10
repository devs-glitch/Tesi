'''
Physical validation: confusion matrix, firing rate analysis, neuron distribution, CV-ISI
'''
import os
import sys
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from xai.sam import load_reference_model
from scripts.training_utils import direct_encode
from scripts.dataloader import build_dataloaders

MANIFEST = 'baseline_manifest.json'
DEAD_THRESHOLD = 0.01
SATURATED_THRESHOLD = 0.80

N_NEURONS_SAMPLE = 200  # per layer -> CV-ISI

def collect_statistics(net, dataloader, time_steps, device, layer_indices=(1, 2, 3, 4)):
    random.seed(42)

    net.eval()

    all_preds, all_labels = [], []

    # Accumulated firing rate per neuron
    per_neuron_sum = {l: None for l in layer_indices}
    n_samples_seen = 0

    # CV-ISI: list of raw spikes train for a fixed sample of neurons position, populated progressively throu the val set
    sampled_positions = {l: None for l in layer_indices}
    isi_pools = {l: None for l in layer_indices}

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            encoded_x = direct_encode(images, time_steps=time_steps)

            spike_out, spike1, spike2, spike3, spike4 = net(encoded_x)
            all_spikes = {1: spike1, 2: spike2, 3: spike3, 4: spike4}

            spike_count = spike_out.sum(dim=0)
            predicted = torch.argmax(spike_count, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            batch_size = images.shape[0]
            n_samples_seen += batch_size

            for layer_index in layer_indices:
                spikes = all_spikes[layer_index]   # [T, B, C, H, W]

                # Firing rate per neuron
                batch_mean = spikes.mean(dim=(0, 1))  # [C, H, W]
                if per_neuron_sum[layer_index] is None:
                    per_neuron_sum[layer_index] = batch_mean * batch_size
                else:
                    per_neuron_sum[layer_index] += batch_mean * batch_size

                # CV-ISI choose positions to sample one at the first batch, then accumulate pike trains from each batch
                C, H, W = spikes.shape[2], spikes.shape[3], spikes.shape[4]
                if sampled_positions[layer_index] is None:
                    all_positions = [(c, h, w) for c in range(C) for h in range(H) for w in range(W)]
                    sampled_positions[layer_index] = random.sample(
                        all_positions, min(N_NEURONS_SAMPLE, len(all_positions))
                    )
                    isi_pools[layer_index] = {pos: [] for pos in sampled_positions[layer_index]}

                # spikes: [T, B, C, H, W] for each image of the batch, for each sample position
                positions = torch.tensor(sampled_positions[layer_index], device=spikes.device)
                # spikes: [T, B, C, H, W] -> trains: [T, B, N_positions]
                trains = spikes[:, :, positions[:, 0], positions[:, 1],
                                positions[:, 2]].cpu().numpy()

                for b in range(batch_size):
                    for n, pos in enumerate(sampled_positions[layer_index]):
                        spike_times = np.where(trains[:, b, n] == 1)[0]
                        if len(spike_times) >= 2:
                            isi_pools[layer_index][pos].extend(np.diff(spike_times).tolist())

    avg_firing_rate_per_neuron = {
        l: (per_neuron_sum[l] / n_samples_seen).cpu().numpy() for l in layer_indices
    }

    return all_preds, all_labels, avg_firing_rate_per_neuron, isi_pools

def plot_confusion_matrix(all_preds, all_labels, class_names):
    # normalized per true class
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format=".2f")
    plt.title("Confusion matrix - normalized per true class")
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_firing_rate_distributions(avg_firing_rate_per_neuron):
    layer_indices = list(avg_firing_rate_per_neuron.keys())
    fig, axes = plt.subplots(1, len(layer_indices), figsize=(5 * len(layer_indices), 4))

    for idx, layer_index in enumerate(layer_indices):
        rates = avg_firing_rate_per_neuron[layer_index].flatten()
        ax = axes[idx]
        ax.hist(rates, bins=50, color="steelblue", edgecolor="black")
        ax.axvline(DEAD_THRESHOLD, color="red", linestyle="--", label=f"dead threshold ({DEAD_THRESHOLD})")
        ax.axvline(SATURATED_THRESHOLD, color="orange", linestyle="--", label=f"saturated threshold ({SATURATED_THRESHOLD})")

        pct_dead = (rates < DEAD_THRESHOLD).mean() * 100
        pct_saturated = (rates > SATURATED_THRESHOLD).mean() * 100

        ax.set_title(f"Layer {layer_index}\n{pct_dead:.1f}% dead, {pct_saturated:.1f}% saturated")
        ax.set_xlabel("Firing rate per neuron")
        ax.legend(fontsize=8)

    fig.suptitle("Firing rate distribution per neuron, per layer")
    plt.savefig("firing_rate_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_cv_isi(isi_pools):
    layer_indices = list(isi_pools.keys())
    fig, axes = plt.subplots(1, len(layer_indices), figsize=(5 * len(layer_indices), 4))

    for idx, layer_index in enumerate(layer_indices):
        cvs = []
        for pos, isi_list in isi_pools[layer_index].items():
            if len(isi_list) >= 5:  # not enough ISI give an unstable estimate
                isi_arr = np.array(isi_list)
                cv = isi_arr.std() / isi_arr.mean() if isi_arr.mean() > 0 else np.nan
                if not np.isnan(cv):
                    cvs.append(cv)

        ax = axes[idx]
        if len(cvs) > 0:
            ax.hist(cvs, bins=30, color="seagreen", edgecolor="black")
            ax.set_title(f"Layer {layer_index} (n={len(cvs)} neurons)")
        else:
            ax.set_title(f"Layer {layer_index}\n(insufficient ISI data)")
        ax.set_xlabel("CV-ISI")

    fig.suptitle(f"CV-ISI per-layer distribution (sample of {N_NEURONS_SAMPLE} neurons/layer,\n"
                 f"ISI aggregated on val set)")
    plt.savefig("cv_isi_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net, time_steps, input_size, _, _ = load_reference_model(device, MANIFEST)
    dataset, _, val_dataloader, _ = build_dataloaders(input_size=input_size, verbose=False)
    class_names = dataset.classes

    print(f'{input_size} px | {len(val_dataloader.dataset)} validation images')

    all_preds, all_labels, avg_firing_rate_per_neuron, isi_pools = collect_statistics(
        net, val_dataloader, time_steps, device, layer_indices=(1, 2, 3, 4)
    )

    plot_confusion_matrix(all_preds, all_labels, class_names)
    plot_firing_rate_distributions(avg_firing_rate_per_neuron)
    plot_cv_isi(isi_pools)


if __name__ == "__main__":
    main()
