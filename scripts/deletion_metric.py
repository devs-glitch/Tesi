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

sys.path.append(os.path.abspath('C:/Users/devam/OneDrive/Tesi'))

from qsnn.sam import load_reference_model, get_layer_spikes, compute_sam
from qsnn.sam_metrics import samples_per_class
from qsnn.training_utils import direct_encode
from qsnn.dataloader import build_dataloaders

MANIFEST = 'baseline_manifest.json'
FRACTIONS = np.arange(0, 1.01, 0.05)
N_PER_CLASS = 15


def get_upsampled_sam_map(net, image, time_steps, device, layer_index, gamma, image_size):
    """
    Calculates SAM aggregated on tiem and mean and it re-dimensions it
    to the resolution of the original image
    """

    spikes = get_layer_spikes(net, image, time_steps, device, layer_index)
    sam_maps = compute_sam(spikes, gamma)        # [T, H, W]
    temporal_agg = sam_maps.mean(dim=0)           # [H, W]

    map_4d = temporal_agg.unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(map_4d, size=(image_size, image_size), mode="bilinear")
    return upsampled.squeeze().cpu()


def get_class_confidence(net, image, time_steps, device, target_class):

    image_batched = image.unsqueeze(0).to(device)
    encoded_x = direct_encode(image_batched, time_steps=time_steps)

    with torch.no_grad():
        spike_out, _, _, _, _ = net(encoded_x)

    spike_count = spike_out.sum(dim=0)            # [1, 4]
    probs = torch.softmax(spike_count, dim=1)
    return probs[0, target_class].item()

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

            target_class = torch.argmax(
                torch.tensor([get_class_confidence(net, image, time_steps, device, c)
                              for c in range(len(class_names))])
            ).item()

            sam_curves.append(deletion_curve(net, image, sam_map, time_steps, device, target_class, "sam"))
            random_curves.append(deletion_curve(net, image, sam_map, time_steps, device, target_class, "random"))

        results[label] = {
            "sam": np.mean(sam_curves, axis=0),
            "random": np.mean(random_curves, axis=0),
        }
        print(f"{class_names[label]}: AUC-SAM={np.trapezoid(results[label]['sam'], FRACTIONS):.4f}, "
              f"AUC-random={np.trapezoid(results[label]['random'], FRACTIONS):.4f}")

    return results

def plot_deletion_curves(results, class_names):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))

    for idx, label in enumerate(sorted(results.keys())):
        ax = axes[idx]
        ax.plot(FRACTIONS, results[label]["sam"], marker="o", label="SAM (importance order)")
        ax.plot(FRACTIONS, results[label]["random"], marker="s", linestyle="--", label="Random")
        ax.set_title(class_names[label])
        ax.set_xlabel("Fraction of removed pixels")
        ax.set_ylabel("Confidence predicted class")
        ax.legend(fontsize=8)

    fig.suptitle(f"Deletion metric - SAM vs random baseline (N={N_PER_CLASS}/class)")
    plt.savefig("deletion_metric.png", dpi=150, bbox_inches="tight")
    plt.show()


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
    
    plot_deletion_curves(results, class_names)


if __name__ == "__main__":
    main()