'''
SAM: Spike Activation Map
'''
import os
import sys
import json
import math

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('C:/Users/devam/OneDrive/Tesi'))
from models.snn_model import GWGlitchSNN
from models.tune import direct_encode


def load_trained_model(device):
    with open("best_hyperparams.json", "r") as f:
        best_params = json.load(f)

    net = GWGlitchSNN(beta=best_params["beta"]).to(device)
    net.load_state_dict(torch.load("best_model.pt", map_location=device))
    net.eval()

    return net, best_params["time_steps"]


def get_layer_spikes(net, image, time_steps, device, layer_index):
    image_batched = image.unsqueeze(0)

    image_batched = image_batched.to(device)
    encoded_x = direct_encode(image_batched, time_steps=time_steps)

    with torch.no_grad():  # niente gradiente, siamo in puro forward/inferenza
        spike_out, spike1, spike2, spike3, spike4 = net(encoded_x)

    spikes_by_layer = {1: spike1, 2: spike2, 3: spike3, 4: spike4}
    selected = spikes_by_layer[layer_index]

    selected_spikes = selected.squeeze(1)

    return selected_spikes

def compute_ncs(spikes, gamma, t):
    """
    spikes: [T, C, H, W], spike train completo del layer
    gamma: iperparametro del kernel esponenziale
    t: il tempo target per cui vuoi calcolare NCS
    ritorna: NCS al tempo t, forma [C, H, W]
    """
    T_total = spikes.shape[0]

    ncs = torch.zeros_like(spikes[0])  # [C, H, W], parte da zero

    for t_prime in range(t + 1):
        tscs = math.exp(-gamma * abs((t - t_prime)))
        ncs += spikes[t_prime] * tscs

    return ncs

def compute_sam(spikes, gamma):
    """
    spikes: [T, C, H, W]
    ritorna: SAM per ogni time-step, forma [T, H, W]
    """
    T_total, C, H, W = spikes.shape

    sam_maps = []

    for t in range(T_total):
        ncs_t = compute_ncs(spikes, gamma, t)  # [C, H, W]

        sam_t = (ncs_t * spikes[t]).sum(dim = 0)

        sam_maps.append(sam_t)

    sam_maps_stacked = torch.stack(sam_maps, dim=0)
    return sam_maps_stacked

def denormalize_image(image, mean, std):
    """
    image: tensore [3, H, W], normalizzato (output diretto del dataloader)
    mean, std: le stesse liste usate in dataloader.py
    ritorna: immagine con valori riportati in [0,1] circa, pronta per imshow
    """
    # TODO 1: converti mean e std (liste di 3 float) in tensori PyTorch di forma
    # [3, 1, 1] — il motivo del reshape a 3 dimensioni: image ha forma [3, H, W],
    # e vuoi che ogni canale (indice 0) venga moltiplicato/sommato per il SUO
    # valore di mean/std, non un valore scalare uguale per tutti i canali
    mean_t = ___
    std_t = ___

    image_denorm = image * std_t + mean_t

    # TODO 2: i valori numerici potrebbero uscire leggermente fuori da [0,1]
    # per via di arrotondamenti — quale funzione torch "schiaccia" un tensore
    # dentro un intervallo min/max senza cambiarne la forma?
    image_denorm = ___

    return image_denorm