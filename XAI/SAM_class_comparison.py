'''
Confronto tra classi: mappe SAM su un campione per ciascuna classe
Class comparison: map SAM on a sample for each class
'''
import os
import sys

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('C:/Users/devam/OneDrive/Tesi'))
from XAI.sam import load_trained_model, get_layer_spikes, compute_sam, denormalize_image
from src_python.dataloader import test_dataloader


def get_one_sample_per_class(test_dataloader, n_classes):
    """
    Scandisce il test_dataloader e ritorna il primo campione trovato per ciascuna classe.
    ritorna: dizionario {classe_idx: tensore immagine [3, H, W]}
    """
    samples = {}

    for images, labels in test_dataloader:
        for i in range(len(labels)):
            label = labels[i].item()

            if label not in samples:
                samples[label] = images[i]

        if len(samples) == n_classes:
            break

    return samples


def compare_classes(net, test_dataloader, time_steps, device, layer_index, gamma, class_names, shared_scale = True):
    n_classes = len(class_names)
    samples = get_one_sample_per_class(test_dataloader, n_classes)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    all_sam_maps = {}
    all_images = {}
    for label, image in samples.items():
        spikes = get_layer_spikes(net, image, time_steps, device, layer_index)
        sam_maps = compute_sam(spikes, gamma)
        all_sam_maps[label] = sam_maps
        all_images[label] = denormalize_image(image, mean, std).permute(1, 2, 0).cpu().numpy()

    if shared_scale:
        global_max = max(m.max().item() for m in all_sam_maps.values())
        global_min = min(m.min().item() for m in all_sam_maps.values())

    t_show = time_steps - 1  

    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))

    for idx, label in enumerate(sorted(samples.keys())):
        ax = axes[idx]
        ax.imshow(all_images[label])
        heatmap = all_sam_maps[label][t_show].cpu().numpy()

        if shared_scale:
            vmin, vmax = global_min, global_max
        else:
            vmin, vmax = heatmap.min(), heatmap.max()

        im = ax.imshow(heatmap, cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax,
                        extent=(0, all_images[label].shape[1], all_images[label].shape[0], 0))
        ax.set_title(f"{class_names[label]}")
        ax.axis("off")

        if not shared_scale:
            fig.colorbar(im, ax=ax, shrink=0.7)

    if shared_scale:
        fig.colorbar(im, ax=axes, shrink=0.6, label="SAM score")    

    suffix = "shared" if shared_scale else "perclass"
    fig.suptitle(f"SAM class comparison - layer {layer_index}, t={t_show}")
    plt.savefig(f"sam_class_comparison_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net, time_steps = load_trained_model(device)
    class_names = test_dataloader.dataset.dataset.classes

    compare_classes(net, test_dataloader, time_steps, device, layer_index=2, gamma=0.5,
                     class_names=class_names, shared_scale=True)
    compare_classes(net, test_dataloader, time_steps, device, layer_index=2, gamma=0.5,
                     class_names=class_names, shared_scale=False)


if __name__ == "__main__":
    main()