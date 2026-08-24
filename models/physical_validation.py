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

sys.path.append(os.path.abspath('C:/Users/devam/OneDrive/Tesi'))
from XAI.sam import load_trained_model, direct_encode
from src_python.dataloader import test_dataloader


# literature based threshold: near-zero firing rate -> dead neuron
# firing rate near al 100% -> saturated/epileptic
DEAD_THRESHOLD = 0.01
SATURATED_THRESHOLD = 0.80

N_NEURONS_SAMPLE = 200  # per layer -> CV-ISI


def collect_test_set_statistics(net, test_dataloader, time_steps, device, layer_indices=(1, 2, 3)):
    net.eval()

    all_preds, all_labels = [], []

    # Accumulated firing rate per neuron: sum over elements of shape [C,H,W<+]
    per_neuron_sum = {l: None for l in layer_indices}
    n_samples_seen = 0

    # CV-ISI: lidt of raw spikes train for a fixed sample per un campione fisso di
    # posizioni neurone, popolate progressivamente attraverso tutto il test set
    sampled_positions = {l: None for l in layer_indices}
    isi_pools = {l: None for l in layer_indices}

    with torch.no_grad():
        for images, labels in test_dataloader:
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

                # --- Firing rate per neurone: accumulo elemento per elemento ---
                # media su tempo e batch di QUESTO batch, sommata all'accumulatore totale
                batch_mean = spikes.mean(dim=(0, 1))  # [C, H, W]
                if per_neuron_sum[layer_index] is None:
                    per_neuron_sum[layer_index] = batch_mean * batch_size
                else:
                    per_neuron_sum[layer_index] += batch_mean * batch_size

                # --- CV-ISI: scegli le posizioni da campionare UNA SOLA VOLTA
                # (al primo batch), poi accumula i loro spike train da ogni batch ---
                C, H, W = spikes.shape[2], spikes.shape[3], spikes.shape[4]
                if sampled_positions[layer_index] is None:
                    all_positions = [(c, h, w) for c in range(C) for h in range(H) for w in range(W)]
                    sampled_positions[layer_index] = random.sample(
                        all_positions, min(N_NEURONS_SAMPLE, len(all_positions))
                    )
                    isi_pools[layer_index] = {pos: [] for pos in sampled_positions[layer_index]}

                # spikes: [T, B, C, H, W] -> per ogni immagine del batch, per ogni
                # posizione campionata, estrai la sequenza temporale [T] e calcola
                # gli indici temporali in cui ha sparato, poi le differenze (ISI)
                for b in range(batch_size):
                    for (c, h, w) in sampled_positions[layer_index]:
                        train = spikes[:, b, c, h, w].cpu().numpy()  # [T], 0/1
                        spike_times = np.where(train == 1)[0]
                        if len(spike_times) >= 2:
                            isi = np.diff(spike_times)
                            isi_pools[layer_index][(c, h, w)].extend(isi.tolist())

    avg_firing_rate_per_neuron = {
        l: (per_neuron_sum[l] / n_samples_seen).cpu().numpy() for l in layer_indices
    }

    return all_preds, all_labels, avg_firing_rate_per_neuron, isi_pools

def plot_confusion_matrix(all_preds, all_labels, class_names):
    # Normalizzata per riga (classe vera), come nel paper originale Gravity Spy
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
        ax.axvline(DEAD_THRESHOLD, color="red", linestyle="--", label=f"soglia morto ({DEAD_THRESHOLD})")
        ax.axvline(SATURATED_THRESHOLD, color="orange", linestyle="--", label=f"soglia saturo ({SATURATED_THRESHOLD})")

        pct_dead = (rates < DEAD_THRESHOLD).mean() * 100
        pct_saturated = (rates > SATURATED_THRESHOLD).mean() * 100

        ax.set_title(f"Layer {layer_index}\n{pct_dead:.1f}% morti, {pct_saturated:.1f}% saturi")
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
            if len(isi_list) >= 5:  # troppo pochi ISI danno una stima instabile
                isi_arr = np.array(isi_list)
                cv = isi_arr.std() / isi_arr.mean() if isi_arr.mean() > 0 else np.nan
                if not np.isnan(cv):
                    cvs.append(cv)

        ax = axes[idx]
        if len(cvs) > 0:
            ax.hist(cvs, bins=30, color="seagreen", edgecolor="black")
            ax.set_title(f"Layer {layer_index} (n={len(cvs)} neuroni)")
        else:
            ax.set_title(f"Layer {layer_index}\n(dati ISI insufficienti)")
        ax.set_xlabel("CV-ISI")

    fig.suptitle(f"Distribuzione CV-ISI per layer (campione di {N_NEURONS_SAMPLE} neuroni/layer,\n"
                 f"ISI aggregati su tutto il test set)")
    plt.savefig("cv_isi_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net, time_steps = load_trained_model(device)
    class_names = test_dataloader.dataset.dataset.classes

    all_preds, all_labels, avg_firing_rate_per_neuron, isi_pools = collect_test_set_statistics(
        net, test_dataloader, time_steps, device, layer_indices=(1, 2, 3, 4)
    )

    plot_confusion_matrix(all_preds, all_labels, class_names)
    plot_firing_rate_distributions(avg_firing_rate_per_neuron)
    plot_cv_isi(isi_pools)


if __name__ == "__main__":
    main()