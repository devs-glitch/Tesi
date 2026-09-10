'''
SAM: Spike Activation Map
'''
import os
import sys
import json
import math

import torch
import matplotlib.pyplot as plt

from model.snn_model import GWGlitchSNN
from scripts.training_utils import direct_encode
from scripts.dataloader import build_dataloaders

def load_reference_model(device, manifest_path = 'baseline_manifest.json', checkpoint_index = 0):
    # load reference model
    with open(manifest_path) as f:
        manifest = json.load(f)

    hp = manifest['hyperparameters']
    net = GWGlitchSNN(beta=hp['beta'], input_size=manifest['input_size']).to(device)
    net.load_state_dict(torch.load(manifest['checkpoints'][checkpoint_index], map_location=device))
    net.eval()

    return net, hp['time_steps'], manifest['input_size'], manifest['sam']['reference_layer'], manifest['sam']['gamma']

def get_layer_spikes(net, image, time_steps, device, layer_index):
    image_batched = image.unsqueeze(0)

    image_batched = image_batched.to(device)
    encoded_x = direct_encode(image_batched, time_steps=time_steps)

    with torch.no_grad(): 
        spike_out, spike1, spike2, spike3, spike4 = net(encoded_x)

    spikes_by_layer = {1: spike1, 2: spike2, 3: spike3, 4: spike4}
    selected = spikes_by_layer[layer_index]

    selected_spikes = selected.squeeze(1)

    return selected_spikes

def compute_ncs(spikes, gamma, t):
    """
    spikes: [T, C, H, W], complete spike train of layer
    gamma: hyperparameter of the exponential kernel
    t: time target of NCS computation
    returns: NCS at time t, shape [C, H, W]
    """

    ncs = torch.zeros_like(spikes[0])

    for t_prime in range(t):
        tscs = math.exp(-gamma * abs((t - t_prime)))
        ncs += spikes[t_prime] * tscs

    return ncs

def compute_sam(spikes, gamma):
    """
    spikes: [T, C, H, W]
    returns: SAM at each time-step, shape [T, H, W]
    """
    T_total, C, H, W = spikes.shape
 
    decay = math.exp(-gamma)
    ncs = torch.zeros_like(spikes[0])
    sam_maps = []
 
    for t in range(T_total):
        sam_t = (ncs * spikes[t]).sum(dim=0)
        sam_maps.append(sam_t)
        ncs = decay * (ncs + spikes[t])
 
    return torch.stack(sam_maps, dim=0)

def compute_sam_reference(spikes, gamma):
    return torch.stack([(compute_ncs(spikes, gamma, t) * spikes[t]).sum(dim=0)
                        for t in range(spikes.shape[0])], dim=0)

def denormalize_image(image, mean, std):
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    image_denorm = image * std_t + mean_t
    image_denorm = torch.clamp(image_denorm, 0, 1)

    return image_denorm

def temporal_centre_of_mass(sam_check):
    # mean timesteps at which the SAM energy concentrates
    # com = sum_t t* E_t / sum_t E_t, with E_t the sapcial energy at step t
    energy = sam_check.sum(dim=(1, 2))
    total = energy.sum()
    if total <= 0:
        return float ('nan')
    steps = torch.arange(sam_check.shape[0], dtype=energy.dtype, device = energy.device)
    return float((steps * energy).sum() / total)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net, time_steps, input_size, layer_index, gamma = load_reference_model(device)

    # take a sample from the val set
    dataset, _, val_dataloader, _ = build_dataloaders(input_size=input_size, verbose=False)

    images, labels = next(iter(val_dataloader))
    sample_image = images[0]
    sample_label = labels[0].item()

    class_names = dataset.classes
    sample_class_name = class_names[sample_label]

    # SAM computation
    spikes = get_layer_spikes(net, sample_image, time_steps, device, layer_index)
    sam_maps = compute_sam(spikes, gamma)

    reference = compute_sam_reference(spikes, gamma)
    max_diff = (sam_maps - reference).abs().max().item()
    assert max_diff < 1e-5, f'recursion and explicit sum diverge: {max_diff}'

    # original image
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_vis = denormalize_image(sample_image, mean, std).permute(1, 2, 0).cpu().numpy()

    vmin, vmax = sam_maps.min().item(), sam_maps.max().item()

    T = sam_maps.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(3 * T, 3))

    for t in range(T):
        ax = axes[t] if T > 1 else axes
        ax.imshow(image_vis)
        heatmap = sam_maps[t].cpu().numpy()
        im = ax.imshow(heatmap, cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax,
                        extent=(0, image_vis.shape[1], image_vis.shape[0], 0))
        ax.set_title(f"t={t}" + (" (no history)" if t == 0 else ""))
        ax.axis("off")

    fig.suptitle(f"SAM - layer {layer_index}, class: {sample_label} ({sample_class_name})")
    fig.colorbar(im, ax=axes, shrink=0.6, label="SAM score")
    plt.savefig("sam_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"SAM computed on layer {layer_index}, {T} time-step, class: {sample_label}")
    print(f"SAM range: [{vmin:.4f}, {vmax:.4f}] | t=0 null by definition")

if __name__ == "__main__":
    main()
