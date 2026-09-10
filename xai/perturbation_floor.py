'''
Sensitivity floor: how much explanation divergence does a negligible weight
perturbation already produce?

noise_floor.py compared models trained from different random seeds and found
spatial agreement as low as PCC 0.153. When we will performs 
the quantisation grid we are asking a different question.
It compares a quantised model with it's own full-precision parent: same initialisation,
same trained weights, perturbed only by rounding
'''

import copy
import json
import os
import statistics as st
import sys

import numpy as np
import torch

from scripts.dataloader import build_dataloaders
from xai.sam import (load_reference_model, get_layer_spikes, compute_sam,
                     temporal_centre_of_mass)
from xai.sam_metrics import samples_per_class, sam_stacks, pearson, nanmean_or_nan
from xai.noise_floor import compare_two_models

MANIFEST = 'baseline_manifest.json'
N_PER_CLASS = 15
OUT_JSON = 'perturbation_floor.json'
CHECKPOINT_INDEX = 0
PERTURBATION_SEED = 999

# Perturbation magnitudes
NOISE_FRACTIONS = [0.01, 0.1, 0.5, 1.0]


def eight_bit_step(net):
    # Quantisation step of a symmetric uniform 8-bit quantiser over the
    
    max_abs = max(p.abs().max().item()
                  for name, p in net.named_parameters() if 'weight' in name)
    return 2.0 * max_abs / (2 ** 8 - 1), max_abs


def perturb_weights(net, sigma, seed):
    # Return a copy of the network with gaussian noise added to its weights

    perturbed = copy.deepcopy(net)
    generator = torch.Generator(device='cpu').manual_seed(seed)

    with torch.no_grad():
        for name, parameter in perturbed.named_parameters():
            if 'weight' not in name:
                continue
            noise = torch.normal(mean=0.0, std=sigma, size=parameter.shape,
                                 generator=generator).to(parameter.device)
            parameter.add_(noise)

    perturbed.eval()
    return perturbed


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)

    gamma = manifest['sam']['gamma']

    layer_index = manifest['sam']['reference_layer']
    input_size = manifest['input_size']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, time_steps, _, _ = load_reference_model(device, MANIFEST,
                                                 checkpoint_index=CHECKPOINT_INDEX)

    step, max_abs = eight_bit_step(net)
    print(f'{input_size} px | layer {layer_index} | gamma {gamma} | T {time_steps}')
    print(f'checkpoint: {manifest["checkpoints"][CHECKPOINT_INDEX]}')
    print(f'max |weight| = {max_abs:.4f}  ->  8-bit step = {step:.6f}\n')

    # every perturbed model see the same images
    dataset, _, val_dataloader, _ = build_dataloaders(input_size=input_size,
                                                      verbose=False)
    classes = dataset.classes
    samples = samples_per_class(val_dataloader, len(classes), N_PER_CLASS)

    # Reference stacks from the unperturbed model
    reference = {c: sam_stacks(net, samples[c], time_steps, device, layer_index,
                               gamma, get_layer_spikes, compute_sam)
                 for c in range(len(classes))}

    results = {}
    for fraction in NOISE_FRACTIONS:
        sigma = fraction * step
        print(f'perturbation {fraction:g} x 8-bit step  (sigma = {sigma:.3e})')

        perturbed = perturb_weights(net, sigma, PERTURBATION_SEED)

        per_class = {}
        for c, class_name in enumerate(classes):
            stacks = sam_stacks(perturbed, samples[c], time_steps, device,
                                layer_index, gamma, get_layer_spikes, compute_sam)
            per_class[class_name] = compare_two_models(reference[c], stacks)

        results[str(fraction)] = {'sigma': sigma, 'per_class': per_class}

        del perturbed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(OUT_JSON, 'w') as f:
        json.dump({'manifest': MANIFEST,
                   'checkpoint_index': CHECKPOINT_INDEX,
                   'gamma': gamma,
                   'layer': layer_index,
                   'input_size': input_size,
                   'n_per_class': N_PER_CLASS,
                   'eight_bit_step': step,
                   'max_abs_weight': max_abs,
                   'perturbation_seed': PERTURBATION_SEED,
                   'noise_fractions': NOISE_FRACTIONS,
                   'results': results}, f, indent=2)
    print(f'\nresults written in {OUT_JSON}\n')

    header = f'{"noise / 8-bit step":<22}' + ''.join(f'{c[:14]:>16}' for c in classes)
    for metric, label in (('spatial_pcc', 'SPATIAL PCC'),
                          ('top_k_iou', 'TOP-20% IoU'),
                          ('com_abs_diff', 'TEMPORAL COM |diff|')):
        print(label)
        print(header)
        print('-' * len(header))
        for fraction in NOISE_FRACTIONS:
            row = f'{fraction:<22g}'
            for class_name in classes:
                row += f'{results[str(fraction)]["per_class"][class_name][metric]:>16.4f}'
            print(row)
        print()


if __name__ == '__main__':
    main()
