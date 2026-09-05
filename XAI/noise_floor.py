'''
FP32-versus-FP32 divergence: the noise floor for every metric in the grid
'''

import itertools
import json
import os
import statistics as st
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath('C:/Users/devam/OneDrive/Tesi'))
from src_python.dataloader import build_dataloaders
from XAI.sam import (load_reference_model, get_layer_spikes, compute_sam,
                     temporal_centre_of_mass)
from XAI.sam_metrics import (samples_per_class, sam_stacks, pearson,
                             nanmean_or_nan, top_k_iou)

MANIFEST = 'baseline_manifest.json'
N_PER_CLASS = 15
OUT_JSON = 'noise_floor.json'
TOP_K = 0.20

# Expected IoU of two independent maps at this k, for reference in the report:
# k / (2 - k). A measured value approaching it means no shared ranking at all.
CHANCE_IOU = TOP_K / (2.0 - TOP_K)


def compare_two_models(stacks_a, stacks_b):
    # All divergence metrics between two models on the same images
    time_aggregated = [(a.mean(dim=0), b.mean(dim=0))
                       for a, b in zip(stacks_a, stacks_b)]

    spatial = nanmean_or_nan([pearson(a, b) for a, b in time_aggregated])
    iou = nanmean_or_nan([top_k_iou(a, b, TOP_K) for a, b in time_aggregated])

    com = nanmean_or_nan([abs(temporal_centre_of_mass(a) - temporal_centre_of_mass(b))
                          for a, b in zip(stacks_a, stacks_b)])

    n_steps = stacks_a[0].shape[0]
    per_step = [nanmean_or_nan([pearson(a[t], b[t])
                                for a, b in zip(stacks_a, stacks_b)])
                for t in range(n_steps)]   # nan at t = 0 by definition

    return {'spatial_pcc': spatial,
            'top_k_iou': iou,
            'com_abs_diff': com,
            'per_step_pcc': per_step}


def aggregate(values):
    # ean, standard deviation and maximum over the seed pairs


    clean = [v for v in values if not np.isnan(v)]
    if not clean:
        return {'mean': float('nan'), 'std': float('nan'), 'max': float('nan')}
    return {'mean': st.mean(clean),
            'std': st.stdev(clean) if len(clean) > 1 else 0.0,
            'max': max(clean),
            'values': clean}


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)

    gamma = manifest['sam']['gamma']

    n_checkpoints = len(manifest['checkpoints'])
    input_size = manifest['input_size']
    layer_index = manifest['sam']['reference_layer']
    print(f'noise floor over {n_checkpoints} FP32 checkpoints | {input_size} px | '
          f'layer {layer_index} | gamma {gamma}')

    #Every model sees identical
    dataset, _, val_dataloader, _ = build_dataloaders(input_size=input_size,
                                                      verbose=False)
    classes = dataset.classes
    samples = samples_per_class(val_dataloader, len(classes), N_PER_CLASS)
    print('samples per class: '
          + str({classes[c]: len(v) for c, v in samples.items()}))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # stacks[checkpoint_index][class_index] = list of [T, H, W] tensors

    stacks = {}
    for k in range(n_checkpoints):
        net, time_steps, _, _ = load_reference_model(device, MANIFEST,
                                                     checkpoint_index=k)
        print(f'  checkpoint {k}: {manifest["checkpoints"][k]}')
        stacks[k] = {c: sam_stacks(net, samples[c], time_steps, device,
                                   layer_index, gamma,
                                   get_layer_spikes, compute_sam)
                     for c in range(len(classes))}
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pairs = list(itertools.combinations(range(n_checkpoints), 2))
    print(f'\nseed pairs compared: {pairs}\n')

    results = {}
    for c, class_name in enumerate(classes):
        per_pair = {f'{i}-{j}': compare_two_models(stacks[i][c], stacks[j][c])
                    for i, j in pairs}

        results[class_name] = {
            'per_pair': per_pair,
            'spatial_pcc': aggregate([p['spatial_pcc'] for p in per_pair.values()]),
            'top_k_iou': aggregate([p['top_k_iou'] for p in per_pair.values()]),
            'com_abs_diff': aggregate([p['com_abs_diff'] for p in per_pair.values()]),
            # worst per-step agreement within each pair, then aggregated:
            # the time step at which two models disagree most
            'worst_step_pcc': aggregate([np.nanmin(p['per_step_pcc'])
                                         for p in per_pair.values()]),
        }

    with open(OUT_JSON, 'w') as f:
        json.dump({'manifest': MANIFEST,
                   'gamma': gamma,
                   'layer': layer_index,
                   'input_size': input_size,
                   'seeds': manifest['seeds'],
                   'n_per_class': N_PER_CLASS,
                   'top_k': TOP_K,
                   'chance_iou': CHANCE_IOU,
                   'classes': results}, f, indent=2)
    print(f'results written to {OUT_JSON}\n')

    header = (f'{"class":<18}{"PCC mean":>11}{"PCC min":>10}'
              f'{"IoU mean":>11}{"IoU min":>10}{"COM mean":>11}{"COM max":>10}')
    print('FP32-vs-FP32 NOISE FLOOR (over seed pairs)')
    print(header)
    print('-' * len(header))
    for class_name in classes:
        r = results[class_name]
        # higher correlation means more agreement, higher centre-of-mass difference means less
        print(f'{class_name:<18}'
              f'{r["spatial_pcc"]["mean"]:>11.4f}{min(r["spatial_pcc"]["values"]):>10.4f}'
              f'{r["top_k_iou"]["mean"]:>11.4f}{min(r["top_k_iou"]["values"]):>10.4f}'
              f'{r["com_abs_diff"]["mean"]:>11.4f}{r["com_abs_diff"]["max"]:>10.4f}')

    print(f'\nchance-level IoU at k={TOP_K}: {CHANCE_IOU:.4f} '
          '(two independent maps)')
    print('\nWorst per-timestep PCC within a pair, by class:')
    for class_name in classes:
        w = results[class_name]['worst_step_pcc']
        print(f'  {class_name:<18} mean {w["mean"]:.4f}   worst {min(w["values"]):.4f}')

if __name__ == '__main__':
    main()