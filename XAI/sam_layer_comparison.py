'''
multi-sample comparison btw layers: mean and std for each layer and each class separately
'''
import os
import sys

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('C:/Users/devam/OneDrive/Tesi'))

from XAI.sam import load_reference_model, get_layer_spikes, compute_sam
from XAI.sam_metrics import samples_per_class
from src_python.dataloader import build_dataloaders

MANIFEST = 'baseline_manifest.json'
# The reference layer is not read from the manifest since this module compares all four layers by construction
LAYER_INDICES = [1, 2, 3, 4]
N_PER_CLASS = 20


def compute_layer_statistics(net, image_list, time_steps, device, layer_index, gamma):
    per_sample_maps = []
    n_channels = None

    for image in image_list:
        spikes = get_layer_spikes(net, image, time_steps, device, layer_index)  # [T, C, H, W]
        n_channels = spikes.shape[1]

        sam_maps = compute_sam(spikes, gamma)          # [T, H, W]
        temporal_agg = sam_maps.mean(dim=0)             # [H, W]
        per_sample_maps.append(temporal_agg)

    stacked = torch.stack(per_sample_maps, dim=0)       # [N, H, W]

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
        axes[0, idx].set_title(f"Layer {layer_index} (C={channel_counts[layer_index]}) - mean")
        axes[0, idx].axis("off")

        im_std = axes[1, idx].imshow(std_maps[layer_index].cpu().numpy(), cmap="magma",
                                       vmin=std_vmin, vmax=std_vmax)
        axes[1, idx].set_title(f"Layer {layer_index} - std dev")
        axes[1, idx].axis("off")

    fig.colorbar(im_mean, ax=axes[0, :], shrink=0.7, label="SAM/channel (mean)")
    fig.colorbar(im_std, ax=axes[1, :], shrink=0.7, label="SAM/channel (std dev)")
    fig.suptitle(f"SAM layer comparison - class: {class_name} (N={N_PER_CLASS} samples)")
    plt.savefig(f"sam_layer_comparison_{class_name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net, time_steps, input_size, _, gamma = load_reference_model(device, MANIFEST)
    dataset, _, val_dataloader, _ = build_dataloaders(input_size=input_size, verbose=False)
    class_names = dataset.classes
    n_classes = len(class_names)

    print(f'{input_size} px | gamma {gamma} | '
          f'{len(val_dataloader.dataset)} validation images')

    samples = samples_per_class(val_dataloader, n_classes, N_PER_CLASS)

    for label in range(n_classes):
        class_name = class_names[label]
        print(f"Layer comparison for class: {class_name}...")
        plot_layer_comparison_for_class(net, samples[label], time_steps, device,
                                        class_name, gamma, LAYER_INDICES)

if __name__ == "__main__":
    main()