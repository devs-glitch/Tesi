'''
Deletion metric (Petsiuk et al., 2018): validazione quantitativa della
correttezza di SAM, con confronto contro una baseline casuale
'''
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('C:/Users/devam/OneDrive/Tesi'))
from XAI.sam import load_trained_model, get_layer_spikes, compute_sam, direct_encode
from XAI.sam_class_comparison import get_n_samples_per_class

FRACTIONS = np.arange(0, 1.01, 0.05)
N_PER_CLASS = 15


def get_upsampled_sam_map(net, image, time_steps, device, layer_index, gamma, image_size=224):
    """
    Calcola SAM (aggregata nel tempo, media) e la ridimensiona alla
    risoluzione dell'immagine originale, così puoi decidere quali PIXEL
    dell'input (non del feature map ridotto) cancellare.
    """
    spikes = get_layer_spikes(net, image, time_steps, device, layer_index)
    sam_maps = compute_sam(spikes, gamma)        # [T, H, W]
    temporal_agg = sam_maps.mean(dim=0)           # [H, W]

    # interpolate si aspetta [B, C, H, W] -> aggiungiamo due dimensioni finte
    map_4d = temporal_agg.unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(map_4d, size=(image_size, image_size), mode="bilinear")
    return upsampled.squeeze()  # torna a [224, 224]


def get_class_confidence(net, image, time_steps, device, target_class):
    """
    Forward pass su una singola immagine, ritorna la probabilità (softmax)
    assegnata dal modello a target_class — la classe predetta sull'immagine
    ORIGINALE, non modificata, che continuiamo a tracciare mentre canceliamo pixel.
    """
    image_batched = image.unsqueeze(0).to(device)
    encoded_x = direct_encode(image_batched, time_steps=time_steps)

    with torch.no_grad():
        spike_out, _, _, _, _ = net(encoded_x)

    spike_count = spike_out.sum(dim=0)            # [1, 4]
    probs = torch.softmax(spike_count, dim=1)
    return probs[0, target_class].item()


def mask_top_fraction(image, importance_map, fraction, order="sam"):
    """
    Cancella (imposta a 0, sostrato neutro nello spazio normalizzato) la
    frazione più importante dei pixel, secondo l'ordine scelto.
    order="sam": i pixel col punteggio SAM più alto vengono cancellati per primi.
    order="random": ordine casuale — è la baseline di controllo.
    """
    H, W = importance_map.shape
    n_pixels = H * W
    n_to_mask = int(fraction * n_pixels)

    flat_importance = importance_map.flatten()

    if order == "sam":
        # TODO: quale funzione torch ordina gli indici di un tensore per
        # valore decrescente, restituendo gli INDICI (non i valori ordinati)?
        # Ti servono i primi n_to_mask indici di questo ordinamento.
        sorted_indices = torch.argsort(flat_importance, descending=True)
    elif order == "random":
        sorted_indices = torch.randperm(n_pixels)

    indices_to_mask = sorted_indices[:n_to_mask]

    mask = torch.ones(n_pixels, dtype=torch.bool)
    mask[indices_to_mask] = False
    mask_2d = mask.reshape(H, W)

    masked_image = image.clone()
    # la maschera vale per la posizione spaziale, applicata a tutti e 3 i canali
    for c in range(3):
        masked_image[c][~mask_2d] = 0.0

    return masked_image


def deletion_curve(net, image, sam_map, time_steps, device, target_class, order):
    confidences = []
    for fraction in FRACTIONS:
        masked = mask_top_fraction(image, sam_map, fraction, order=order)
        conf = get_class_confidence(net, masked, time_steps, device, target_class)
        confidences.append(conf)
    return confidences


def run_deletion_analysis(net, samples, time_steps, device, layer_index, gamma, class_names):
    results = {}  # {classe: {"sam": [curve1, curve2, ...], "random": [...]}}

    for label, image_list in samples.items():
        sam_curves, random_curves = [], []

        for image in image_list:
            sam_map = get_upsampled_sam_map(net, image, time_steps, device, layer_index, gamma)

            image_gpu = image.to(device)
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
        ax.plot(FRACTIONS, results[label]["sam"], marker="o", label="SAM (ordine importanza)")
        ax.plot(FRACTIONS, results[label]["random"], marker="s", linestyle="--", label="Random")
        ax.set_title(class_names[label])
        ax.set_xlabel("Frazione pixel rimossi")
        ax.set_ylabel("Confidenza classe predetta")
        ax.legend(fontsize=8)

    fig.suptitle(f"Deletion metric — SAM vs baseline casuale (N={N_PER_CLASS}/classe)")
    plt.savefig("deletion_metric.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src_python.old_dataloader import test_dataloader

    net, time_steps = load_trained_model(device)
    class_names = test_dataloader.dataset.dataset.classes
    n_classes = len(class_names)

    samples = get_n_samples_per_class(test_dataloader, n_classes, N_PER_CLASS)

    results = run_deletion_analysis(net, samples, time_steps, device,
                                     layer_index=2, gamma=0.5, class_names=class_names)

    plot_deletion_curves(results, class_names)


if __name__ == "__main__":
    main()