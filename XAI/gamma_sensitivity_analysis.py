'''
gamma sensitivity analysis
'''
import os
import sys
import numpy as np

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('C:/Users/devam/OneDrive/Tesi'))
from XAI.SAM_class_comparison import compute_sam_statistics_for_class

GAMMA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.5, 3.0]

def gamma_sensitivity_analysis(net, image_list, time_steps, device, layer_index, gamma_values):
    maps_per_gamma = {}

    for gamma in gamma_values:
        mean_map, _ = compute_sam_statistics_for_class(net, image_list, time_steps, device, layer_index, gamma)
        maps_per_gamma[gamma] = mean_map

    n = len(gamma_values)
    corr_matrix = np.zeros((n, n))

    for i, g1 in enumerate(gamma_values):
        for j, g2 in enumerate(gamma_values):
            map1 = maps_per_gamma[g1].cpu().numpy().flatten()
            map2 = maps_per_gamma[g2].cpu().numpy().flatten()
            corr_matrix[i, j] = np.corrcoef(map1, map2)[0, 1]

    return maps_per_gamma, corr_matrix

def plot_gamma_sensitivity(maps_per_gamma, corr_matrix, gamma_values, class_name, layer_index):
    n = len(gamma_values)

    # Riga 1: le mappe medie per ciascun gamma, scala condivisa (stesso layer,
    # stessa classe — numericamente comparabili, come nel confronto tra classi)
    all_values = torch.cat([m.flatten() for m in maps_per_gamma.values()])
    vmin, vmax = all_values.min().item(), all_values.max().item()

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    for idx, gamma in enumerate(gamma_values):
        heatmap = maps_per_gamma[gamma].cpu().numpy()
        im = axes[idx].imshow(heatmap, cmap="jet", vmin=vmin, vmax=vmax)
        axes[idx].set_title(f"γ={gamma}")
        axes[idx].axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label="SAM score (media)")
    fig.suptitle(f"Sensibilità a gamma — classe: {class_name}, layer {layer_index}")
    plt.savefig(f"gamma_sensitivity_maps_{class_name}.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Riga 2: la matrice di correlazione tra tutte le coppie di gamma
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im2 = ax2.imshow(corr_matrix, cmap="viridis", vmin=0, vmax=1)
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(gamma_values)
    ax2.set_yticklabels(gamma_values)
    ax2.set_xlabel("gamma")
    ax2.set_ylabel("gamma")

    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f"{corr_matrix[i, j]:.4f}", ha="center", va="center",
                      color="white" if corr_matrix[i, j] < 0.7 else "black", fontsize=8)

    fig2.colorbar(im2, label="Correlazione di Pearson")
    fig2.suptitle(f"Correlazione tra mappe SAM a gamma diversi — classe: {class_name}")
    plt.savefig(f"gamma_sensitivity_correlation_{class_name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    from XAI.sam import load_trained_model
    from XAI.SAM_class_comparison import get_n_samples_per_class
    from src_python.dataloader import test_dataloader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, time_steps = load_trained_model(device)
    class_names = test_dataloader.dataset.dataset.classes

    n_classes = len(class_names)
    samples = get_n_samples_per_class(test_dataloader, n_classes, n_per_class=15)

    # Estesa a tutte le classi (non solo Blip) per rigore statistico completo —
    # accettiamo il costo computazionale maggiore (7 gamma x 15 immagini x 4 classi)
    for target_class_idx in range(n_classes):
        class_name = class_names[target_class_idx]
        image_list = samples[target_class_idx]

        print(f"Analisi sensibilità gamma per classe: {class_name}...")

        maps_per_gamma, corr_matrix = gamma_sensitivity_analysis(
            net, image_list, time_steps, device, layer_index=2, gamma_values=GAMMA_VALUES
        )

        plot_gamma_sensitivity(maps_per_gamma, corr_matrix, GAMMA_VALUES, class_name, layer_index=2)


if __name__ == "__main__":
    main()