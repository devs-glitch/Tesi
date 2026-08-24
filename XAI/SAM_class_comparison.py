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


def get_n_samples_per_class(test_dataloader, n_classes, n_per_class):
    """
    Raccoglie fino a n_per_class campioni per ciascuna classe.
    ritorna: {classe_idx: [lista di tensori immagine]}
    """
    samples = {c: [] for c in range(n_classes)}

    for images, labels in test_dataloader:
        for i in range(len(labels)):
            label = labels[i].item()
            if len(samples[label]) < n_per_class:
                samples[label].append(images[i])

        # TODO 1: esci dal ciclo esterno quando OGNI classe ha raggiunto
        # n_per_class campioni — non solo quando una sola ce l'ha fatta.
        # Indizio: serve verificare la condizione su TUTTI i valori del
        # dizionario "samples", non su uno solo — quale funzione Python
        # applicata a un'espressione generatore fa esattamente questo?
        if all(len(v) >= n_per_class for v in samples.values()):
            break

    return samples


def compute_sam_statistics_for_class(net, image_list, time_steps, device, layer_index, gamma):
    """
    Per ogni immagine: calcola SAM, aggrega nel tempo (media su T).
    Poi calcola media e deviazione standard TRA i campioni della classe.
    ritorna: mean_map [H,W], std_map [H,W]
    """
    per_sample_maps = []

    for image in image_list:
        spikes = get_layer_spikes(net, image, time_steps, device, layer_index)
        sam_maps = compute_sam(spikes, gamma)          # [T, H, W]

        # TODO 2: aggregazione temporale — media lungo l'asse T (dim=0),
        # ottenendo una mappa singola [H, W] per QUESTA immagine
        temporal_agg = sam_maps.mean(dim=0)
        per_sample_maps.append(temporal_agg)

    # TODO 3: impila i 25 tensori [H,W] in un unico tensore [N,H,W]
    # (stesso strumento usato più volte in sam.py per unire liste di tensori)
    stacked = torch.stack(per_sample_maps, dim=0)

    # TODO 4: calcola media e deviazione standard LUNGO L'ASSE DEI CAMPIONI
    # (dim=0, quello con N elementi) — non lungo H o W, che devono restare intatti
    mean_map = stacked.mean(dim=0)
    std_map = stacked.std(dim=0)

    return mean_map, std_map

def compute_cv_map(mean_map, std_map, epsilon=1e-6):
    """
    Coefficiente di variazione: std/mean, con epsilon per evitare divisioni
    per (quasi) zero nelle zone dove il modello non si attiva affatto —
    lì il rapporto non ha significato interpretativo, è solo rumore numerico.
    """
    cv_map = std_map / (mean_map + epsilon)
    return cv_map

def plot_class_statistics(net, test_dataloader, time_steps, device, layer_index, gamma, class_names, n_per_class):
    n_classes = len(class_names)
    samples = get_n_samples_per_class(test_dataloader, n_classes, n_per_class)

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

    # Il CV grezzo ha una coda lunghissima (valori enormi dove mean≈0, anche
    # con epsilon). Per una visualizzazione leggibile, tronchiamo la scala
    # al 95° percentile invece che al massimo assoluto — pratica statistica
    # standard per non far dominare la colorbar da pochi outlier numerici
    # privi di significato interpretativo. Va dichiarato esplicitamente
    # in tesi come scelta di visualizzazione, non di calcolo: i valori
    # grezzi restano intatti, solo la scala colore è limitata.
    all_cv_values = torch.cat([m.flatten() for m in cv_maps.values()])
    cv_vmax = torch.quantile(all_cv_values, 0.95).item()

    fig, axes = plt.subplots(3, n_classes, figsize=(4 * n_classes, 12))

    for idx, label in enumerate(sorted(samples.keys())):
        im_mean = axes[0, idx].imshow(mean_maps[label].cpu().numpy(), cmap="jet",
                                        vmin=mean_vmin, vmax=mean_vmax)
        axes[0, idx].set_title(f"{class_names[label]} — media")
        axes[0, idx].axis("off")

        im_std = axes[1, idx].imshow(std_maps[label].cpu().numpy(), cmap="magma",
                                       vmin=std_vmin, vmax=std_vmax)
        axes[1, idx].set_title(f"{class_names[label]} — dev. std")
        axes[1, idx].axis("off")

        im_cv = axes[2, idx].imshow(cv_maps[label].cpu().numpy(), cmap="viridis",
                                      vmin=0, vmax=cv_vmax)
        axes[2, idx].set_title(f"{class_names[label]} — CV")
        axes[2, idx].axis("off")

    fig.colorbar(im_mean, ax=axes[0, :], shrink=0.7, label="SAM score (media)")
    fig.colorbar(im_std, ax=axes[1, :], shrink=0.7, label="SAM score (dev. std)")
    fig.colorbar(im_cv, ax=axes[2, :], shrink=0.7, label="Coefficiente di variazione")

    fig.suptitle(f"SAM per classe (N={n_per_class} campioni/classe) — layer {layer_index}\n"
                 f"CV troncato al 95° percentile per leggibilità")
    plt.savefig(f"sam_class_statistics_layer{layer_index}.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net, time_steps = load_trained_model(device)
    class_names = test_dataloader.dataset.dataset.classes

    plot_class_statistics(net, test_dataloader, time_steps, device,
                          layer_index=2, gamma=0.5, class_names=class_names, n_per_class=25)


if __name__ == "__main__":
    main()