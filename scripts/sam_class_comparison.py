'''
Class comparison: map SAM on a sample for each class
'''

import os
import sys

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('C:/Users/devam/OneDrive/Tesi'))
from qsnn.sam import (load_reference_model, get_layer_spikes, compute_sam)
from qsnn.sam_metrics import samples_per_class
from qsnn.dataloader import build_dataloaders

MANIFEST = 'baseline_manifest.json'
N_PER_CLASS = 25


def compute_sam_statistics_for_class(net, image_list, time_steps, device, layer_index, gamma):
    """
    calculate SAM for each image, aggregating on time, then calculating
    mean and std between samples of a class. Returns mean_map and std_map
    """
    per_sample_maps = []

    for image in image_list:
        spikes = get_layer_spikes(net, image, time_steps, device, layer_index)
        sam_maps = compute_sam(spikes, gamma)          # [T, H, W]

        temporal_agg = sam_maps.mean(dim=0)
        per_sample_maps.append(temporal_agg)

    stacked = torch.stack(per_sample_maps, dim=0)

    mean_map = stacked.mean(dim=0)
    std_map = stacked.std(dim=0)

    return mean_map, std_map

def compute_cv_map(mean_map, std_map, epsilon=1e-6):
    # coefficient of variation (std/mean)
    cv_map = std_map / (mean_map + epsilon)
    return cv_map

def plot_class_statistics(net, val_dataloader, time_steps, device, layer_index, gamma, class_names, n_per_class):
    n_classes = len(class_names)
    samples = samples_per_class(val_dataloader, n_classes, n_per_class)

    mean_maps, std_maps, cv_maps = {}, {}, {}
    for label, image_list in samples.items():
        mean_map, std_map = compute_sam_statistics_for_class(
            net, image_list, time_steps, device, layer_index, gamma
        )
        mean_maps[label] = mean_map
        std_maps[label] = std_map
        cv_maps[label] = compute_cv_map(mean_map, std_map)

    mean_vmax = max(m.max().item() for m in mean_maps.values())
    mean_vmin = min(m.min().item() for m in mean_maps.values())
    std_vmax = max(m.max().item() for m in std_maps.values())
    std_vmin = min(m.min().item() for m in std_maps.values())

    all_cv_values = torch.cat([m.flatten() for m in cv_maps.values()])
    cv_vmax = torch.quantile(all_cv_values, 0.95).item()

    fig, axes = plt.subplots(3, n_classes, figsize=(4 * n_classes, 12))

    for idx, label in enumerate(sorted(samples.keys())):
        im_mean = axes[0, idx].imshow(mean_maps[label].cpu().numpy(), cmap="jet",
                                        vmin=mean_vmin, vmax=mean_vmax)
        axes[0, idx].set_title(f"{class_names[label]} - mean")
        axes[0, idx].axis("off")

        im_std = axes[1, idx].imshow(std_maps[label].cpu().numpy(), cmap="magma",
                                       vmin=std_vmin, vmax=std_vmax)
        axes[1, idx].set_title(f"{class_names[label]} - std dev")
        axes[1, idx].axis("off")

        im_cv = axes[2, idx].imshow(cv_maps[label].cpu().numpy(), cmap="viridis",
                                      vmin=0, vmax=cv_vmax)
        axes[2, idx].set_title(f"{class_names[label]} - CV")
        axes[2, idx].axis("off")

    fig.colorbar(im_mean, ax=axes[0, :], shrink=0.7, label="SAM score (mean)")
    fig.colorbar(im_std, ax=axes[1, :], shrink=0.7, label="SAM score (std dev)")
    fig.colorbar(im_cv, ax=axes[2, :], shrink=0.7, label="Coefficient of variation")

    fig.suptitle(f"SAM for class (N={n_per_class} samples/class) - layer {layer_index}\n"
                 f"CV truncated at 95th percentile")
    plt.savefig(f"sam_class_statistics_layer{layer_index}.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net, time_steps, input_size, layer_index, gamma = load_reference_model(device, MANIFEST)
    dataset, _, val_dataloader, _ = build_dataloaders(input_size=input_size, verbose=False)
    class_names = dataset.classes

    print(f'{input_size} px | layer {layer_index} | gamma {gamma} | '
          f'{len(val_dataloader.dataset)} validation images')

    plot_class_statistics(net, val_dataloader, time_steps, device,
                          layer_index=layer_index, gamma=gamma,
                          class_names=class_names, n_per_class=N_PER_CLASS)


if __name__ == "__main__":
    main()