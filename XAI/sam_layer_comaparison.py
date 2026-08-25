'''
Confronto tra layer, multi-campione: media e deviazione standard per
ciascun layer, per ciascuna classe separatamente
'''
import os
import sys

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('C:/Users/devam/OneDrive/Tesi'))
from XAI.sam import load_trained_model, get_layer_spikes, compute_sam
from XAI.sam_class_comparison import get_n_samples_per_class

LAYER_INDICES = [1, 2, 3, 4]
N_PER_CLASS = 20
GAMMA = 0.5


def compute_layer_statistics(net, image_list, time_steps, device, layer_index, gamma):
    """
    Come compute_sam_statistics_for_class, ma con l'aggiunta della
    normalizzazione per numero di canali — necessaria qui perché stiamo
    confrontando layer con profondità diverse (16, 32, 64 canali),
    a differenza del confronto tra classi dove il layer era sempre lo stesso.
    """
    per_sample_maps = []
    n_channels = None

    for image in image_list:
        spikes = get_layer_spikes(net, image, time_steps, device, layer_index)  # [T, C, H, W]
        n_channels = spikes.shape[1]

        sam_maps = compute_sam(spikes, gamma)          # [T, H, W]
        temporal_agg = sam_maps.mean(dim=0)             # [H, W]
        per_sample_maps.append(temporal_agg)

    stacked = torch.stack(per_sample_maps, dim=0)       # [N, H, W]

    # TODO: normalizza sia mean_map che std_map dividendo per n_channels —
    # stesso principio già applicato nel vecchio sam_layer_comparison.py
    # a singolo campione, qui applicato però DOPO aver calcolato mean/std
    # sui campioni. Attenzione: la normalizzazione va fatta su entrambe
    # le mappe risultanti (mean e std), non solo su una
    mean_map = stacked.mean(dim=0)
    std_map = stacked.std(dim=0)

    mean_map_normalized = mean_map / n_channels
    std_map_normalized = std_map / n_channels

    return mean_map_normalized, std_map_normalized, n_channels


def plot_layer_comparison_for_class(net, image_list, time_steps, device, class_name, gamma, layer_indices):
    mean_maps, std_maps, channel_counts = {}, {}, {}

    for layer_index in layer_indices:
        mean_map, std_map, n_channels = compute_layer_statistics(
            net, image_list, time_steps, device, layer_index, gamma
        )
        mean_maps[layer_index] = mean_map
        std_maps[layer_index] = std_map
        channel_counts[layer_index] = n_channels

    mean_vmax = max(m.max().item() for m in mean_maps.values())
    mean_vmin = min(m.min().item() for m in mean_maps.values())
    std_vmax = max(m.max().item() for m in std_maps.values())
    std_vmin = min(m.min().item() for m in std_maps.values())

    n = len(layer_indices)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    for idx, layer_index in enumerate(layer_indices):
        im_mean = axes[0, idx].imshow(mean_maps[layer_index].cpu().numpy(), cmap="jet",
                                        vmin=mean_vmin, vmax=mean_vmax)
        axes[0, idx].set_title(f"Layer {layer_index} (C={channel_counts[layer_index]}) — media")
        axes[0, idx].axis("off")

        im_std = axes[1, idx].imshow(std_maps[layer_index].cpu().numpy(), cmap="magma",
                                       vmin=std_vmin, vmax=std_vmax)
        axes[1, idx].set_title(f"Layer {layer_index} — dev. std")
        axes[1, idx].axis("off")

    fig.colorbar(im_mean, ax=axes[0, :], shrink=0.7, label="SAM/canale (media)")
    fig.colorbar(im_std, ax=axes[1, :], shrink=0.7, label="SAM/canale (dev. std)")
    fig.suptitle(f"Confronto SAM tra layer — classe: {class_name} (N={N_PER_CLASS} campioni)")
    plt.savefig(f"sam_layer_comparison_{class_name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src_python.old_dataloader import test_dataloader

    net, time_steps = load_trained_model(device)
    class_names = test_dataloader.dataset.dataset.classes
    n_classes = len(class_names)

    samples = get_n_samples_per_class(test_dataloader, n_classes, N_PER_CLASS)

    for label in range(n_classes):
        class_name = class_names[label]
        print(f"Confronto layer per classe: {class_name}...")
        plot_layer_comparison_for_class(net, samples[label], time_steps, device,
                                          class_name, GAMMA, LAYER_INDICES)


if __name__ == "__main__":
    main()