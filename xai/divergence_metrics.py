'''
Measure of how far the SAM maps of a quantised model move from the FP32 reference
'''

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from model.quantization import apply_quantisation
from model.snn_model import GWGlitchSNN
from scripts.dataloader import build_dataloaders
from xai.sam import (MANIFEST, load_reference_model, get_layer_spikes,
                     compute_sam, temporal_centre_of_mass)
from xai.sam_metrics import samples_per_class, sam_stacks, pearson, top_k_iou

OUT_JSON = Path('results/divergence.json')
RUNS_DIR = Path('runs')

N_PER_CLASS = 20
TOP_K = 0.20

FLOOR = {'pcc': 0.975, 'iou': 0.704, 'com': 0.029}
INDEPENDENT = {'pcc': 0.153, 'iou': 0.234, 'com': 0.115}
IOU_CHANCE = TOP_K / (2 - TOP_K)


def _ssim(map_a, map_b):
    # SSIM on two 2-D maps, each min-max normalised to [0, 1] first

    from skimage.metrics import structural_similarity

    def normalise(x):
        x = x.numpy().astype(np.float64)
        span = x.max() - x.min()
        if span == 0:
            return None
        return (x - x.min()) / span

    a, b = normalise(map_a), normalise(map_b)
    if a is None or b is None:
        return float('nan')
    return float(structural_similarity(a, b, data_range=1.0))

# Loading

def load_quantised_run(run_dir, device):
    # Load a fine-tuned model from its run directory

    with open(run_dir / 'run.json', encoding='utf-8') as f:
        summary = json.load(f)
    with open(MANIFEST, encoding='utf-8') as f:
        manifest = json.load(f)

    hp = manifest['hyperparameters']
    net = GWGlitchSNN(beta=hp['beta'], input_size=manifest['input_size']).to(device)
    net.load_state_dict(torch.load(run_dir / 'best_model.pt', map_location=device))
    net.eval()

    membrane_bits = summary['quantisation'].get('membrane_bits')
    membrane_range = tuple(summary.get('membrane_range', (0.0, 2.0)))
    handle = apply_quantisation(net, membrane_bits=membrane_bits,
                                membrane_range=membrane_range)
    return net, handle, summary


def load_manifest_checkpoint(index, device):
    # one of the FP32 checkpoints named in the manifest for the self-test
    return load_reference_model(device, MANIFEST, checkpoint_index=index)

# comparison
def compare_stacks(stack_a, stack_b, use_ssim=True):
    # every metric for one pair of SAM stacks [T, H, W] from the same image
    aggregate_a = stack_a.sum(dim=0)
    aggregate_b = stack_b.sum(dim=0)

    per_step = [pearson(stack_a[t], stack_b[t]) for t in range(stack_a.shape[0])]

    return {
        'pcc': pearson(aggregate_a, aggregate_b),
        'iou': top_k_iou(aggregate_a, aggregate_b, k=TOP_K),
        'ssim': _ssim(aggregate_a, aggregate_b) if use_ssim else float('nan'),
        'per_step_pcc': per_step,
        'com_a': temporal_centre_of_mass(stack_a),
        'com_b': temporal_centre_of_mass(stack_b),
    }


def compare_models(net_a, net_b, samples, class_names, time_steps, device,
                   layer_index, gamma, use_ssim=True):
    # Per-class divergence between two models on identical images
    results = {}
    for label, images in samples.items():
        if not images:
            continue
        stacks_a = sam_stacks(net_a, images, time_steps, device, layer_index, gamma)
        stacks_b = sam_stacks(net_b, images, time_steps, device, layer_index, gamma)

        per_image = [compare_stacks(a, b, use_ssim)
                     for a, b in zip(stacks_a, stacks_b)]

        per_step = np.array([m['per_step_pcc'] for m in per_image], dtype=np.float64)
        com_shift = [m['com_b'] - m['com_a'] for m in per_image]

        with np.errstate(invalid='ignore'):
            results[class_names[label]] = {
                'n': len(per_image),
                'pcc': float(np.nanmean([m['pcc'] for m in per_image])),
                'pcc_std': float(np.nanstd([m['pcc'] for m in per_image])),
                'iou': float(np.nanmean([m['iou'] for m in per_image])),
                'ssim': float(np.nanmean([m['ssim'] for m in per_image])),
                'per_step_pcc': [float(v) for v in np.nanmean(per_step, axis=0)],
                'com_shift': float(np.nanmean(com_shift)),
                'com_shift_abs': float(np.nanmean(np.abs(com_shift))),
            }
    return results


# Reporting
def verdict(pcc):
    # ->where a correlation sits between the two measured extremes
    if pcc >= FLOOR['pcc']:
        return 'at the numerical floor'
    if pcc <= INDEPENDENT['pcc']:
        return 'at the indipendent-solution level'
    span = (pcc - INDEPENDENT['pcc']) / (FLOOR['pcc'] - INDEPENDENT['pcc'])
    return f'{100 * span:.0f}% of the way from independent to identical'


def print_comparison(title, results):
    print(f'\n{title}')
    header = (f'{"class":<18}{"PCC":>8}{"+-":>7}{"IoU":>8}{"SSIM":>8}'
              f'{"CoM shift":>11}{"worst t":>9}')
    print(header)
    print('-' * len(header))
    for name, r in results.items():
        steps = np.array(r['per_step_pcc'], dtype=np.float64)
        worst = int(np.nanargmin(steps)) if not np.all(np.isnan(steps)) else -1
        print(f'{name:<18}{r["pcc"]:>8.3f}{r["pcc_std"]:>7.3f}{r["iou"]:>8.3f}'
              f'{r["ssim"]:>8.3f}{r["com_shift"]:>+11.3f}'
              f'{worst if worst >= 0 else "-":>9}')

    pccs = [r['pcc'] for r in results.values()]
    print(f'  worst class {min(pccs):.3f} - {verdict(min(pccs))}')


def run_self_test(device, samples, class_names, time_steps, layer_index, gamma,
                  use_ssim=True):
    # compare the manifest's FP32 checkpoints against each other
    with open(MANIFEST, encoding='utf-8') as f:
        n_checkpoints = len(json.load(f)['checkpoints'])

    if n_checkpoints < 2:
        print('self-test needs at least two FP32 checkpoints in the manifest')
        return {}

    net_a, *_ = load_manifest_checkpoint(0, device)
    out = {}
    for index in range(1, n_checkpoints):
        net_b, *_ = load_manifest_checkpoint(index, device)
        results = compare_models(net_a, net_b, samples, class_names, time_steps,
                                 device, layer_index, gamma, use_ssim)
        out[f'fp32_0_vs_{index}'] = results
        print_comparison(f'Self-test: FP32 checkpoint 0 vs {index}', results)

    worst = min(r['pcc'] for results in out.values() for r in results.values())
    print(f'\n  expected: a worst class near PCC {INDEPENDENT["pcc"]:.3f} '
          f'(independent solutions)')
    print(f'  measured: {worst:.3f}')
    if abs(worst - INDEPENDENT['pcc']) > 0.05:
        print('  Mismatch: pipeline does not reproduce a number it was')
    else:
        print('  Matches: pipeline reproduces its own calibration')
    return out


def main():
    parser = argparse.ArgumentParser(
        description='SAM divergence between quantised models and the FP32 reference')
    parser.add_argument('--n-per-class', type=int, default=N_PER_CLASS)
    parser.add_argument('--self-test', action='store_true',
                        help='compare the FP32 checkpoints against each other')
    parser.add_argument('--no-ssim', action='store_true',
                        help='skip SSIM (redundant with PCC and IoU)')
    parser.add_argument('--runs', type=str, nargs='+', default=None,
                        help='run directory names. default: every qat_* run present')
    args = parser.parse_args()

    use_ssim = not args.no_ssim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reference, time_steps, input_size, layer_index, gamma = load_reference_model(
        device, MANIFEST)
    dataset, _, val_dataloader, _ = build_dataloaders(input_size=input_size,
                                                      verbose=False)
    class_names = dataset.classes

    samples = samples_per_class(val_dataloader, len(class_names), args.n_per_class)

    print(f'{input_size} px | T={time_steps} | layer {layer_index} | gamma {gamma}')
    print(f'{args.n_per_class} images per class, top-k {TOP_K:g} '
          f'(chance IoU {IOU_CHANCE:.3f})')

    output = {'manifest': str(MANIFEST), 'layer': layer_index, 'gamma': gamma,
              'n_per_class': args.n_per_class, 'top_k': TOP_K,
              'floor': FLOOR, 'independent': INDEPENDENT}

    output['self_test'] = run_self_test(device, samples, class_names, time_steps,
                                        layer_index, gamma, use_ssim)
    if args.self_test:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        print(f'\nwritten to {OUT_JSON}')
        return

    if args.runs:
        run_dirs = [RUNS_DIR / name for name in args.runs]
    else:
        run_dirs = sorted(d for d in RUNS_DIR.glob('qat_*')
                          if (d / 'run.json').exists())

    if not run_dirs:
        print('\nno qat_* runs found - nothing to compare yet')
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        return

    output['runs'] = {}
    for run_dir in run_dirs:
        net, handle, summary = load_quantised_run(run_dir, device)
        results = compare_models(reference, net, samples, class_names, time_steps,
                                 device, layer_index, gamma, use_ssim)
        handle.remove()

        output['runs'][run_dir.name] = {
            'arm': summary['arm'],
            'weight_bits': summary['quantisation'].get('weight_bits'),
            'membrane_bits': summary['quantisation'].get('membrane_bits'),
            'seed': summary['seed'],
            'qat_accuracy': summary['best_val_accuracy'],
            'divergence': results,
        }
        print_comparison(f'{run_dir.name}  ({summary["arm"]}, '
                         f'accuracy {summary["best_val_accuracy"]:.4f})', results)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f'\n{"=" * 70}')
    print(f'  PCC {FLOOR["pcc"]:.3f} / IoU {FLOOR["iou"]:.3f} - the numerical floor')
    print(f'  PCC {INDEPENDENT["pcc"]:.3f} / IoU {INDEPENDENT["iou"]:.3f}')
    print(f'  IoU {IOU_CHANCE:.3f} is chance for top-{TOP_K:g}')
    print(f'\n{"=" * 70}')

    print(f'\nwritten to {OUT_JSON}')


if __name__ == '__main__':
    main()