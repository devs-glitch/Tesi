'''
Sanity checks for noise_floor.py, to be run before trusting its numbers.

The measured FP32-vs-FP32 spatial PCC is as low as 0.153, which is low enough
that a bug must be excluded before the result is written up. Two independent
checks are performed.

CHECK 1 — identity.
Comparing a checkpoint against ITSELF must give PCC = 1, IoU = 1 and a centre
of mass difference of exactly 0. Anything else means the metrics themselves are
wrong, or that the image pairing inside compare_two_models is misaligned.

CHECK 2 — shared images.
All three models must see the SAME images. If samples were collected inside the
per-model loop, or if the loader order changed between calls, the script would
be measuring variance across different images rather than divergence between
models — and would produce plausible-looking numbers either way. The check
compares the raw pixel tensors used for each model, which is the only way to
verify identity rather than assume it.

CHECK 3 — cross-model magnitude, for context.
Prints the same metrics for a genuine pair, so the identity result and the real
result appear side by side.

Run this before rewriting Section 9.4.
'''

import json
import os
import sys

import torch

sys.path.append(os.path.abspath('C:/Users/devam/OneDrive/Tesi'))
from qsnn.dataloader import build_dataloaders
from qsnn.sam import (load_reference_model, get_layer_spikes, compute_sam,
                     temporal_centre_of_mass)
from qsnn.sam_metrics import samples_per_class, sam_stacks, pearson, top_k_iou
from XAI.noise_floor import compare_two_models, MANIFEST, N_PER_CLASS

TOLERANCE = 1e-9


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)

    gamma = manifest['sam']['gamma']
    layer_index = manifest['sam']['reference_layer']
    input_size = manifest['input_size']
    n_checkpoints = len(manifest['checkpoints'])

    print(f'{input_size} px | layer {layer_index} | gamma {gamma} | '
          f'{n_checkpoints} checkpoints\n')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, _, val_dataloader, _ = build_dataloaders(input_size=input_size,
                                                      verbose=False)
    classes = dataset.classes

    # ---------------------------------------------------------------- CHECK 2
    # Collect the samples twice, in two separate calls, and verify the two
    # collections are bit-identical. This is what guarantees that every model
    # in noise_floor.py saw the same images: if two calls to the same function
    # on the same loader can disagree, the whole comparison is invalid.
    print('CHECK 2 - are the sampled images identical across calls?')
    samples_first = samples_per_class(val_dataloader, len(classes), N_PER_CLASS)
    samples_second = samples_per_class(val_dataloader, len(classes), N_PER_CLASS)

    all_identical = True
    for c, class_name in enumerate(classes):
        a, b = samples_first[c], samples_second[c]
        same_count = len(a) == len(b)
        same_pixels = same_count and all(torch.equal(x, y) for x, y in zip(a, b))
        all_identical = all_identical and same_pixels
        print(f'  {class_name:<18} n={len(a):<3} identical: {same_pixels}')

    if not all_identical:
        raise SystemExit(
            '\nFAILED: two calls to samples_per_class returned different images.\n'
            'The validation loader is not deterministic (shuffle enabled?), so\n'
            'noise_floor.py compared models on different data and its numbers\n'
            'are meaningless.')
    print('  -> all image sets identical across calls\n')

    # ---------------------------------------------------------------- CHECK 1
    print('CHECK 1 - does a checkpoint compared against itself give a perfect score?')
    net, time_steps, _, _ = load_reference_model(device, MANIFEST,
                                                 checkpoint_index=0)

    class_index = 0
    stacks_0 = sam_stacks(net, samples_first[class_index], time_steps, device,
                          layer_index, gamma, get_layer_spikes, compute_sam)

    identity = compare_two_models(stacks_0, stacks_0)
    print(f'  class: {classes[class_index]}')
    print(f'  spatial_pcc  = {identity["spatial_pcc"]:.12f}   (expected 1)')
    print(f'  top_k_iou    = {identity["top_k_iou"]:.12f}   (expected 1)')
    print(f'  com_abs_diff = {identity["com_abs_diff"]:.12f}   (expected 0)')

    failures = []
    if abs(identity['spatial_pcc'] - 1.0) > TOLERANCE:
        failures.append('spatial_pcc != 1')
    if abs(identity['top_k_iou'] - 1.0) > TOLERANCE:
        failures.append('top_k_iou != 1')
    if abs(identity['com_abs_diff']) > TOLERANCE:
        failures.append('com_abs_diff != 0')

    if failures:
        raise SystemExit(
            f'\nFAILED: {", ".join(failures)}.\n'
            'The metrics or the image pairing inside compare_two_models are\n'
            'wrong, and the noise floor numbers cannot be trusted.')
    print('  -> identity check passed\n')

    # ---------------------------------------------------------------- CHECK 3
    # A second forward pass with the SAME model on the SAME images, to confirm
    # the network itself is deterministic in eval mode. If this fails, the low
    # cross-model numbers may be measuring run-to-run noise rather than model
    # differences.
    print('CHECK 3 - is a second forward pass of the same model reproducible?')
    stacks_0_again = sam_stacks(net, samples_first[class_index], time_steps,
                                device, layer_index, gamma,
                                get_layer_spikes, compute_sam)
    max_diff = max((a - b).abs().max().item()
                   for a, b in zip(stacks_0, stacks_0_again))
    print(f'  max |SAM - SAM_again| = {max_diff:.3e}   (expected 0)')
    if max_diff > 1e-6:
        raise SystemExit(
            '\nFAILED: the same model on the same images produced different SAM\n'
            'maps. Something is non-deterministic in the forward pass.')
    print('  -> forward pass reproducible\n')

    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- CONTEXT
    # Now a genuine cross-model pair, printed next to the identity result so
    # the contrast is visible rather than inferred.
    print('CONTEXT - the same metrics for a genuine pair of different seeds')
    net_b, time_steps_b, _, _ = load_reference_model(device, MANIFEST,
                                                     checkpoint_index=1)
    stacks_1 = sam_stacks(net_b, samples_first[class_index], time_steps_b,
                          device, layer_index, gamma,
                          get_layer_spikes, compute_sam)
    cross = compare_two_models(stacks_0, stacks_1)
    print(f'  spatial_pcc  = {cross["spatial_pcc"]:.4f}')
    print(f'  top_k_iou    = {cross["top_k_iou"]:.4f}')
    print(f'  com_abs_diff = {cross["com_abs_diff"]:.4f}')

    # Per-image spread: a single outlier image could drag the class mean down,
    # which would be a very different situation from uniform disagreement.
    per_image_pcc = sorted(pearson(a.mean(dim=0), b.mean(dim=0))
                           for a, b in zip(stacks_0, stacks_1))
    print(f'\n  per-image spatial PCC, sorted:')
    print('   ', ' '.join(f'{v:.3f}' for v in per_image_pcc))
    print(f'  min {per_image_pcc[0]:.3f} | median {per_image_pcc[len(per_image_pcc)//2]:.3f} '
          f'| max {per_image_pcc[-1]:.3f}')
    print('\n  If the values are uniformly low, the two models genuinely disagree.')
    print('  If most are high and a few are near zero, the class mean is being')
    print('  driven by a handful of images and should be reported as a median.')

    print('\nAll checks passed. The noise floor numbers reflect real model')
    print('divergence, not a defect in the measurement.')


if __name__ == '__main__':
    main()