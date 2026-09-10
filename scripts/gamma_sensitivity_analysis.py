'''
Gamma sensitivity of SAM
We compute:
- The spatial PCC between time-aggregated maps
- temporal center of mass of the SAM energy, which is the axis gamma acts on
- Per-timestep PCC against a reference gamma

If the centre of mass shifts appreciably with gamma
while the spatial correlation stays high, gamma governs WHEN the
explanation concentrates and not WHERE, and the choice of gamma must be taken
considering the temporal axis, which is the axis on which quantization
divergence is measured
'''

import json
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('C:/Users/devam/OneDrive/Tesi'))

from qsnn.dataloader import build_dataloaders
from qsnn.sam import load_reference_model, get_layer_spikes, compute_sam, temporal_centre_of_mass
from qsnn.sam_metrics import samples_per_class, sam_stacks, pearson, nanmean_or_nan

GAMMA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.5, 3.0]
MANIFEST = 'baseline_manifest.json'
N_PER_CLASS = 15
OUT_JSON = 'gamma_sensitivity.json'


def analyse_class(net, image_list, time_steps, device, layer_index, gammas):
    # Run the three measurements for one class.

    stacks = {g: sam_stacks(net, image_list, time_steps, device, layer_index, g) for g in gammas}
    n= len(gammas)

    spatial = np.zeros((n, n))
    for i, g1 in enumerate(gammas):
        for j, g2 in enumerate(gammas):
            per_image = [pearson(s1.mean(dim=0), s2.mean(dim=0))
                for s1, s2 in zip(stacks[g1], stacks[g2])]
            spatial[i, j] = nanmean_or_nan(per_image)

    com = {g: nanmean_or_nan([temporal_centre_of_mass(s) for s in stacks[g]]) for g in gammas}

    reference_gamma = gammas[len(gammas) // 2]
    n_steps = stacks[reference_gamma][0].shape[0]
    per_step = {}
    for g in gammas:
        curve = []
        for t in range(n_steps):
            per_image = [pearson(s_ref[t], s_g[t])
                         for s_ref, s_g in zip(stacks[reference_gamma], stacks[g])]
            # nan at t=0 for every gamma by definition of the NCS
            curve.append(nanmean_or_nan(per_image))
        per_step[g] = curve

    return spatial, com, per_step, reference_gamma, _

def plot_class(spatial, com, per_step, gammas, reference_gamma,
               class_name, layer_index):
    # Three-panel figure for one class
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.5))
 
    # Panel 1 - spatial PCC matrix
    im = ax1.imshow(spatial, cmap='viridis', vmin=0, vmax=1)
    ax1.set_xticks(range(len(gammas)))
    ax1.set_yticks(range(len(gammas)))
    ax1.set_xticklabels(gammas)
    ax1.set_yticklabels(gammas)
    ax1.set_xlabel('gamma')
    ax1.set_ylabel('gamma')
    ax1.set_title('spatial PCC (time-aggregated)')
    for i in range(len(gammas)):
        for j in range(len(gammas)):
            ax1.text(j, i, f'{spatial[i, j]:.3f}', ha='center', va='center',
                     fontsize=7,
                     color='white' if spatial[i, j] < 0.7 else 'black')
    fig.colorbar(im, ax=ax1, shrink=0.8)
 
    # Panel 2 - where in time the explanation concentrates
    ax2.plot(gammas, [com[g] for g in gammas], 'o-')
    ax2.set_xlabel('gamma')
    ax2.set_ylabel('temporal centre of mass (time steps)')
    ax2.set_title('when the explanation concentrates')
    ax2.grid(alpha=0.3)
 
    # Panel 3 - a flat, high curve means the temporal structure survives
    # a curve decaying with t means late-stage reasoning diverges while early
    # processing is preserved
    for g in gammas:
        ax3.plot(range(len(per_step[g])), per_step[g], 'o-',
                 label=f'gamma={g}', alpha=0.8)
    ax3.set_xlabel('time step')
    ax3.set_ylabel(f'PCC vs gamma={reference_gamma}')
    ax3.set_title('per-timestep agreement')
    ax3.set_ylim(0, 1.02)
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)
 
    fig.suptitle(f'Gamma sensitivity - {class_name}, layer {layer_index}')
    plt.savefig(f'gamma_sensitivity_{class_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net, time_steps, input_size, layer_index = load_reference_model(device, MANIFEST)
    print(f"reference model: {input_size}px | T={time_steps} | SAM layer={layer_index}")

    dataset, _, val_dataloader, _ = build_dataloaders(input_size=input_size, verbose=False)

    classes = dataset.classes
    samples= samples_per_class(val_dataloader, len(classes), N_PER_CLASS)
    print('samples per class: ' + str({classes[c]: len(v) for c, v in samples.items()}))

    results = {}
    for class_index, class_name in enumerate(classes):
        print(f'analysing {class_name}...')
        spatial, com, per_step, reference_gamma = analyse_class(net, samples[class_index], time_steps, device, layer_index, GAMMA_VALUES)
        plot_class(spatial, com, per_step, GAMMA_VALUES, reference_gamma, class_name, layer_index)
        results[class_name] = {
            'spatial_pcc': spatial.tolist(),
            'temporal_com': com,
            'per_step_pcc_vs_reference': per_step,
            'reference_gamma': reference_gamma,
        }

    with open(OUT_JSON, 'w') as f:
        json.dump({'gammas': GAMMA_VALUES,
                    'layer': layer_index,
                    'input_size': input_size,
                    'time_steps': time_steps,
                    'n_per_class': N_PER_CLASS,
                    'classes': results}, f, indent=2
                   )
    print(f'\nresults written in {OUT_JSON}\n')

    print('Temporal center of mass per gamma (time_steps)')
    print(f'{"class":<18}' + ''.join(f'{"g=" + str(g):>9}' for g in GAMMA_VALUES))
    for class_name in classes:
        com = results[class_name]['temporal_com']
        print(f'{class_name:<18}' + ''.join(f'{com[g]:>9.3f}' for g in GAMMA_VALUES))
 
    print(f'\nSpatial PCC between the extreme gammas '
          f'({GAMMA_VALUES[0]} vs {GAMMA_VALUES[-1]}), and matrix minimum')
    for class_name in classes:
        s = np.array(results[class_name]['spatial_pcc'])
        print(f'  {class_name:<18} extremes {s[0, -1]:.4f}   minimum {s.min():.4f}')

if __name__ == '__main__':
    main()