"""
Single sample SAM visualisation: one panel per time step
"""

import matplotlib.pyplot as plt
import torch

from scripts.dataloader import build_dataloaders
from xai.sam import (MANIFEST, load_reference_model, get_layer_spikes,
                     compute_sam, compute_sam_reference, denormalize_image)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OUT_PNG = 'results/Figures/Model_and_Explainability/sam/sam_visualization.png'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, time_steps, input_size, layer_index, gamma = load_reference_model(
        device, MANIFEST)
    print(f'{input_size} px | layer {layer_index} | gamma {gamma} | T={time_steps}')

    dataset, _, val_dataloader, _ = build_dataloaders(input_size=input_size,
                                                      verbose=False)
    images, labels = next(iter(val_dataloader))
    sample_image, sample_label = images[0], labels[0].item()
    class_name = dataset.classes[sample_label]

    spikes = get_layer_spikes(net, sample_image, time_steps, device, layer_index)
    sam_maps = compute_sam(spikes, gamma)

    # The O(T) recursion must equal the O(T^2) definition

    reference = compute_sam_reference(spikes, gamma)
    max_diff = (sam_maps - reference).abs().max().item()
    assert max_diff < 1e-5, f'recursion and explicit sum diverge: {max_diff}'

    image_vis = denormalize_image(sample_image, IMAGENET_MEAN, IMAGENET_STD)
    image_vis = image_vis.permute(1, 2, 0).cpu().numpy()

    # shared colour scale across time steps
    vmin, vmax = sam_maps.min().item(), sam_maps.max().item()

    T = sam_maps.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(3 * T, 3))

    for t in range(T):
        ax = axes[t] if T > 1 else axes
        ax.imshow(image_vis)
        im = ax.imshow(sam_maps[t].cpu().numpy(), cmap='jet', alpha=0.5,
                       vmin=vmin, vmax=vmax,
                       extent=(0, image_vis.shape[1], image_vis.shape[0], 0))
        ax.set_title(f't={t}' + (' (no history)' if t == 0 else ''))
        ax.axis('off')

    fig.suptitle(f'SAM - layer {layer_index}, class: {class_name} (gamma={gamma})')
    fig.colorbar(im, ax=axes, shrink=0.6, label='SAM score')
    plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'SAM range: [{vmin:.4f}, {vmax:.4f}] | t=0 null by definition')
    print(f'figure written to {OUT_PNG}')


if __name__ == '__main__':
    main()