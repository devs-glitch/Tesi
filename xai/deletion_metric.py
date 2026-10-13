'''
Deletion metric: quantitative validation of the correctness of SAM with a comparison to random baseline
'''
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random


from xai.sam import load_reference_model, get_layer_spikes, compute_sam
from xai.sam_metrics import samples_per_class
from scripts.training_utils import direct_encode
from scripts.dataloader import build_dataloaders

from scripts.figure_style import (BLUE, BLUE_FILL, INK, MUTED,
                                  FIG_DIR, clean_axes, frame, save,
                                  use_thesis_style)

MANIFEST = 'config/baseline_manifest.json'
FRACTIONS = np.arange(0, 1.01, 0.05)
N_PER_CLASS = 15

use_thesis_style()


def plot_deletion_curves(results, class_names, fractions, n_per_class,
                         out_dir=FIG_DIR):
    labels = sorted(results)
    chance = 1.0 / len(class_names)

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.6), sharex=True, sharey=True)
    axes = axes.flat

    for ax, label in zip(axes, labels):
        sam = np.asarray(results[label]['sam'], dtype=float)
        random_order = np.asarray(results[label]['random'], dtype=float)
        delta = float(np.trapezoid(sam, fractions)
                      - np.trapezoid(random_order, fractions))

        # The metric is the area between the two curves, so the figure shades it
        ax.fill_between(fractions, sam, random_order, color=BLUE_FILL,
                        linewidth=0, zorder=2)
        ax.plot(fractions, random_order, color=MUTED, linestyle=(0, (4, 2)),
                linewidth=1.4, marker='s', markersize=3, zorder=3,
                label='random order')
        ax.plot(fractions, sam, color=BLUE, linewidth=1.6, marker='o',
                markersize=3, zorder=4, label='SAM order')
        ax.axhline(chance, color=INK, linestyle=(0, (1, 2)), linewidth=1,
                   zorder=1)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.04)
        clean_axes(ax)
        # The sign carries the reading so it is printed
        ax.set_title(f'{class_names[label].replace("_", " ")}\n'
                     f'$\\Delta = {delta:+.3f}$', fontsize=8.4, pad=7,
                     linespacing=1.5)

    axes[0].set_ylabel('confidence in the predicted class', fontsize=8.5)
    fig.supxlabel('fraction of pixels removed', fontsize=8.5, y=0.015)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper right',
               bbox_to_anchor=(0.995, 1 - 0.06 / fig.get_figheight()),
               frameon=False, fontsize=8, ncol=2, handlelength=2.4)

    frame(fig, 'Deletion metric, SAM order against random order',
          note=f'{n_per_class} validation images per class, mean curve. '
               f'$\\Delta$ is the area under the SAM curve minus the area under '
               f'the random curve: negative means the map is faithful. '
               f'Dotted line: chance for {len(class_names)} classes, '
               f'{chance:.2f}.')
    save(fig, 'deletion_metric', out_dir)


def get_upsampled_sam_map(net, image, time_steps, device, layer_index, gamma, image_size):
    # Calculates SAM aggregated on tiem and mean and it re-dimensions it to the resolution of the original image

    spikes = get_layer_spikes(net, image, time_steps, device, layer_index)
    sam_maps = compute_sam(spikes, gamma)        # [T, H, W]
    temporal_agg = sam_maps.mean(dim=0)           # [H, W]

    map_4d = temporal_agg.unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(map_4d, size=(image_size, image_size), mode="bilinear")
    return upsampled.squeeze().cpu()


def class_probabilities(net, image, time_steps, device):
    # softmax over the per-class spike counts of one image, as a vector

    image_batched = image.unsqueeze(0).to(device)
    encoded_x = direct_encode(image_batched, time_steps=time_steps)

    with torch.no_grad():
        spike_out, _, _, _, _ = net(encoded_x)

    spike_count = spike_out.sum(dim=0)            # [1, 4]
    return torch.softmax(spike_count, dim=1)[0]   # [4]


def get_class_confidence(net, image, time_steps, device, target_class):

    return class_probabilities(net, image, time_steps, device)[target_class].item()

def mask_top_fraction(image, importance_map, fraction, ordering):
    # Set to zero the given fraction of pixels in the order supplied

    H, W = importance_map.shape
    n_pixels = H * W
    n_to_mask = int(fraction * n_pixels)

    indices_to_mask = ordering[:n_to_mask]

    mask = torch.zeros(n_pixels, dtype=torch.bool)
    mask[indices_to_mask] = True
    mask_2d = mask.reshape(H, W)

    masked_image = image.clone()

    # the spatial mask applies to all three channels
    for c in range(3):
        masked_image[c][mask_2d] = 0.0

    return masked_image

def deletion_curve(net, image, sam_map, time_steps, device, target_class, order):
    n_pixels = sam_map.numel()
    if order == "sam":
        ordering = torch.argsort(sam_map.flatten(), descending=True)
    else:
        ordering = torch.randperm(n_pixels)

    confidences = []
    for fraction in FRACTIONS:
        masked = mask_top_fraction(image, sam_map, fraction, ordering)
        conf = get_class_confidence(net, masked, time_steps, device, target_class)
        confidences.append(conf)
    return confidences


def run_deletion_analysis(net, samples, time_steps, device, layer_index, gamma, class_names, image_size):
    results = {}  # {class: {"sam": [curve1, curve2, ...], "random": [...]}}

    for label, image_list in samples.items():
        sam_curves, random_curves = [], []

        for image in image_list:
            sam_map = get_upsampled_sam_map(net, image, time_steps, device, layer_index, gamma, image_size)

            # One forward pass
            target_class = int(
                class_probabilities(net, image, time_steps, device).argmax()
            )

            sam_curves.append(deletion_curve(net, image, sam_map, time_steps, device, target_class, "sam"))
            random_curves.append(deletion_curve(net, image, sam_map, time_steps, device, target_class, "random"))

        results[label] = {
            "sam": np.mean(sam_curves, axis=0),
            "random": np.mean(random_curves, axis=0),
        }
        print(f"{class_names[label]}: AUC-SAM={np.trapezoid(results[label]['sam'], FRACTIONS):.4f}, "
              f"AUC-random={np.trapezoid(results[label]['random'], FRACTIONS):.4f}")

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    net, time_steps, input_size, layer_index, gamma = load_reference_model(device, MANIFEST)
    dataset, _, val_dataloader, _ = build_dataloaders(input_size=input_size, verbose=False)
    class_names = dataset.classes
    n_classes = len(class_names)

    print(f'{input_size} px | layer {layer_index} | gamma {gamma} | '
          f'{len(val_dataloader.dataset)} validation images')

    samples = samples_per_class(val_dataloader, n_classes, N_PER_CLASS)

    results = run_deletion_analysis(net, samples, time_steps, device,
                                    layer_index=layer_index, gamma=gamma,
                                    class_names=class_names, image_size=input_size)
    
    plot_deletion_curves(results, class_names, FRACTIONS, N_PER_CLASS)


if __name__ == "__main__":
    main()