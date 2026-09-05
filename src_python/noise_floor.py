'''
FP32-versus-FP32 divergence: the noise floor for every metric in the grid
'''

import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath('C:/Users/devam/OneDrive/Tesi'))
# TODO E0 — imports.
#   From XAI.sam:         load_reference_model, temporal_centre_of_mass
#   From XAI.sam_metrics: samples_per_class, sam_stacks, pearson,
#                         nanmean_or_nan, top_k_iou
#   From src_python.dataloader: build_dataloaders
#
#   Note load_reference_model takes a checkpoint_index argument: that is how
#   you load seed 2 and seed 3 rather than always the first one.
from XAI.sam import load_reference_model, temporal_centre_of_mass, samples_per_class, sam_stacks, pearson, nanmean_or_nan, top_k_iou
from src_python


MANIFEST = 'baseline_manifest.json'
N_PER_CLASS = 15
OUT_JSON = 'noise_floor.json'
TOP_K = 0.20


def compare_two_models(stacks_a, stacks_b):
    '''All divergence metrics between two models on the SAME images.

    TODO E1 — implement.

    stacks_a and stacks_b are lists of SAM stacks [T, H, W], one per image,
    produced by two different checkpoints on the SAME image list and at the
    SAME gamma. That correspondence is the whole point: zip(stacks_a, stacks_b)
    must pair image i with image i. If the two lists came from different image
    sets you would be measuring image variance, not model divergence.

    Compute, per image, then average across images with nanmean_or_nan:

      'spatial_pcc'   — pearson between the two time-aggregated maps
                        (s.mean(dim=0) for each). Scale-invariant, so the fact
                        that the two models produce SAM values of different
                        magnitude does not matter here.

      'top_k_iou'     — top_k_iou between the two time-aggregated maps.
                        Rank-based, so also insensitive to scale.

      'com_abs_diff'  — absolute difference of temporal_centre_of_mass between
                        the two stacks, in time steps. A ratio of sums, so
                        again scale-free. Report the ABSOLUTE difference: the
                        sign would depend on the arbitrary ordering of the two
                        seeds and would average towards zero across pairs,
                        hiding the magnitude.

      'per_step_pcc'  — a list of length T: pearson between the two maps at
                        each time step, averaged over images. Expect nan at
                        t = 0, where both maps are identically zero.

    Returns: dict with those four keys.
    '''
    pass


def main():
    '''TODO E2 — assemble.

    STRUCTURE

    1. Read the manifest yourself (json.load) to get the number of checkpoints
       and the frozen gamma. Do not hard-code either: the manifest is the
       single source of truth, and gamma is now filled in.

       Guard: if manifest['sam']['gamma'] is None, raise with a clear message.
       Running this before gamma is frozen would produce numbers that cannot be
       compared with anything computed later.

    2. Build the validation dataloader once, at the manifest input size, and
       collect the samples ONCE. Every model must see the identical images —
       this is the single most important requirement in the whole script.
       Collecting them inside a per-model loop would silently break it if the
       loader order ever changed.

    3. Load each checkpoint in turn (checkpoint_index = 0, 1, 2) and compute
       its SAM stacks for every class. Store them as
       stacks[seed_index][class_index] = list of [T, H, W] tensors.

       Memory: three models x four classes x fifteen images x [8, 28, 28] is
       about 4 MB in total on CPU. Holding them all is fine, and it avoids
       recomputing SAM three times for the three pairs.

    4. For each class, and for each of the three seed PAIRS — use
       itertools.combinations(range(n_checkpoints), 2), which gives (0,1),
       (0,2), (1,2) — call compare_two_models and store the result.

    5. Aggregate: for each class and each metric, report mean, standard
       deviation and MAXIMUM across the three pairs. The maximum matters more
       than the mean here, because the noise floor is a bound: what you need to
       claim later is 'quantisation divergence exceeds anything two full
       precision models produce', and that is a statement about the worst case.

    6. Write everything to OUT_JSON, including gamma, layer, input size and the
       seed list, so the file is self-describing.

    7. Print a table: one row per class, columns for spatial PCC, top-k IoU and
       COM difference, each as mean and max across pairs. Then a final row
       aggregating over classes.

    WHAT TO EXPECT, AND WHAT EACH OUTCOME MEANS

      If spatial PCC between seeds comes out around 0.99, the spatial metric is
      stable across initialisation and can detect quantisation effects well
      below that level. If it comes out at 0.90, then only large quantisation
      effects will be distinguishable, and that limit must be stated.

      Watch the class ordering. The gamma analysis found Violin_Mode 57 times
      more sensitive than Extremely_Loud. If the seed noise floor shows the
      same ordering, then class sensitivity is a property of the class
      morphology rather than of gamma, which is a finding in its own right and
      means the grid results must be reported per class.

      The COM difference is the number to watch most closely. Changing gamma by
      a factor of thirty moves it by about 0.9 steps. If two seeds already
      differ by a comparable amount, the temporal centre of mass is too noisy
      to serve as a divergence metric and the per-timestep curve must carry
      that part of the argument instead.
    '''
    pass


if __name__ == '__main__':
    main()