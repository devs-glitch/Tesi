'''
Assess validity of the explanations of the quantized model via three modes:

fp32 / own      the FP32 reference with its own map = the baseline

quantised / own   the quantised model with its own map

quantised / fp32  the quantised model with the reference's map
'''

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.dataloader import build_dataloaders
from scripts.training_utils import direct_encode
from xai.divergence_metrics import load_quantised_run, load_fp32_for_seed
from xai.sam import MANIFEST, load_reference_model, get_layer_spikes, compute_sam
from xai.sam_metrics import samples_per_class

OUT_JSON = Path('results/cross_faithfulness.json')
RUNS_DIR = Path('runs')

N_PER_CLASS = 10
FRACTIONS = np.round(np.arange(0.0, 1.01, 0.1), 2)

# A class is interpretable when the FP32 baseline beats random deletion by at
# least this much in area under the curve
MIN_BASELINE_GAP = 0.05


# --------------------------------------------------------------------------- #
def upsampled_sam_map(net, image, time_steps, device, layer_index, gamma, size):
    # SAM aggregated over time and resized to the input resolution

    spikes = get_layer_spikes(net, image, time_steps, device, layer_index)
    aggregated = compute_sam(spikes, gamma).mean(dim=0)
    resized = F.interpolate(aggregated.unsqueeze(0).unsqueeze(0),
                            size=(size, size), mode='bilinear',
                            align_corners=False)
    return resized.squeeze().cpu()


def confidence(net, image, time_steps, device, target_class):
    # Softmax probability the model assigns to target_class
    with torch.no_grad():
        spike_out, *_ = net(direct_encode(image.unsqueeze(0).to(device), time_steps))
    return float(torch.softmax(spike_out.sum(dim=0), dim=1)[0, target_class])


def predicted_class(net, image, time_steps, device):
    with torch.no_grad():
        spike_out, *_ = net(direct_encode(image.unsqueeze(0).to(device), time_steps))
    return int(spike_out.sum(dim=0).argmax(dim=1))


def mask_top_fraction(image, order_indices, fraction):
    # blank the given fraction of pixels, highest-ranked first
    n_pixels = order_indices.numel()
    n_masked = int(fraction * n_pixels)
    if n_masked == 0:
        return image

    masked = image.clone()
    flat = masked.reshape(image.shape[0], -1)
    flat[:, order_indices[:n_masked]] = 0.0
    return flat.reshape(image.shape)


def deletion_curve(net, image, order_indices, time_steps, device, target_class):
    return [confidence(net, mask_top_fraction(image, order_indices, f),
                       time_steps, device, target_class)
            for f in FRACTIONS]


def auc(curve):
    return float(np.trapezoid(curve, FRACTIONS))


def evaluate_model(net, samples, maps, time_steps, device, targets, seed=0):
    # deletion AUC with the given maps and the random baseline

    generator = torch.Generator().manual_seed(seed)
    per_class = {}

    for label, images in samples.items():
        sam_aucs, random_aucs = [], []
        for index, image in enumerate(images):
            target = targets[label][index]

            order = torch.argsort(maps[label][index].flatten(), descending=True)
            sam_aucs.append(auc(deletion_curve(net, image, order, time_steps,
                                               device, target)))

            shuffled = torch.randperm(order.numel(), generator=generator)
            random_aucs.append(auc(deletion_curve(net, image, shuffled, time_steps,
                                                  device, target)))

        per_class[label] = {
            'auc_sam': float(np.mean(sam_aucs)),
            'auc_random': float(np.mean(random_aucs)),
            'gap': float(np.mean(random_aucs) - np.mean(sam_aucs)),
            'n': len(images),
        }
    return per_class


def compute_maps(net, samples, time_steps, device, layer_index, gamma, size):
    return {label: [upsampled_sam_map(net, img, time_steps, device, layer_index,
                                      gamma, size) for img in images]
            for label, images in samples.items()}


def main():
    parser = argparse.ArgumentParser(
        description='Deletion-metric faithfulness of quantised models and their maps')
    parser.add_argument('--n-per-class', type=int, default=N_PER_CLASS)
    parser.add_argument('--runs', type=str, nargs='+', default=None,
                        help='run directory names; default: every qat_* run')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reference, time_steps, input_size, layer_index, gamma = load_reference_model(
        device, MANIFEST)
    dataset, _, val_dataloader, _ = build_dataloaders(input_size=input_size,
                                                      verbose=False)
    class_names = dataset.classes
    samples = samples_per_class(val_dataloader, len(class_names), args.n_per_class)

    print(f'{input_size} px | T={time_steps} | layer {layer_index} | gamma {gamma}')
    print(f'{args.n_per_class} images per class | {len(FRACTIONS)} deletion steps')

    # baseline

    # The class each curve tracks fixed by the reference for every mode
    targets = {label: [predicted_class(reference, img, time_steps, device)
                       for img in images]
               for label, images in samples.items()}

    print('\nbaseline: FP32 reference with its own maps')
    reference_maps = compute_maps(reference, samples, time_steps, device,
                                  layer_index, gamma, input_size)
    baseline = evaluate_model(reference, samples, reference_maps, time_steps,
                              device, targets)

    interpretable = {}
    print(f'{"class":<18}{"AUC-SAM":>10}{"AUC-random":>12}{"gap":>9}{"":>4}')
    for label, r in baseline.items():
        ok = r['gap'] >= MIN_BASELINE_GAP
        interpretable[label] = ok
        print(f'{class_names[label]:<18}{r["auc_sam"]:>10.4f}{r["auc_random"]:>12.4f}'
              f'{r["gap"]:>9.4f}{"" if ok else "   no signal":>4}')

    usable = [class_names[l] for l, ok in interpretable.items() if ok]
    if not usable:
        raise SystemExit(
            'No class has a baseline gap above the threshold')
    print(f'\ninterpretable: {", ".join(usable)}')

    #  runs
    run_dirs = ([RUNS_DIR / name for name in args.runs] if args.runs else
                sorted(d for d in RUNS_DIR.glob('qat_*') if (d / 'run.json').exists()))

    output = {'layer': layer_index, 'gamma': gamma, 'n_per_class': args.n_per_class,
              'fractions': FRACTIONS.tolist(),
              'min_baseline_gap': MIN_BASELINE_GAP,
              'interpretable': {class_names[l]: ok for l, ok in interpretable.items()},
              'baseline': {class_names[l]: r for l, r in baseline.items()},
              'runs': {}}

    for run_dir in run_dirs:
        net, handle, summary = load_quantised_run(run_dir, device)
        seed_reference = load_fp32_for_seed(summary['seed'], input_size, device)

        seed_maps = compute_maps(seed_reference, samples, time_steps, device,
                                 layer_index, gamma, input_size)

        seed_targets = {label: [predicted_class(seed_reference, img, time_steps, device)
                                for img in images]
                        for label, images in samples.items()}
        seed_baseline = evaluate_model(seed_reference, samples, seed_maps,
                                       time_steps, device, seed_targets)

        own_maps = compute_maps(net, samples, time_steps, device, layer_index,
                                        gamma, input_size)

        own = evaluate_model(net, samples, own_maps, time_steps, device, seed_targets)
        borrowed = evaluate_model(net, samples, seed_maps, time_steps, device, seed_targets)

        handle.remove()

        output['runs'][run_dir.name] = {
            'arm': summary['arm'],
            'membrane_bits': summary['quantisation'].get('membrane_bits'),
            'weight_bits': summary['quantisation'].get('weight_bits'),
            'seed': summary['seed'],
            'own_maps': {class_names[l]: r for l, r in own.items()},
            'reference_maps': {class_names[l]: r for l, r in borrowed.items()},
            'baseline': {class_names[l]: r for l, r in seed_baseline.items()},
        }

        print(f'\n{run_dir.name}  ({summary["arm"]})')
        print(f'{"class":<18}{"baseline gap":>14}{"own map":>12}{"FP32 map":>12}')
        for label in sorted(baseline):
            mark = '' if interpretable[label] else '  (no signal)'
            print(f'{class_names[label]:<18}{seed_baseline[label]["gap"]:>14.4f}'
                  f'{own[label]["gap"]:>12.4f}{borrowed[label]["gap"]:>12.4f}{mark}')

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f'\nwritten to {OUT_JSON}')


if __name__ == '__main__':
    main()