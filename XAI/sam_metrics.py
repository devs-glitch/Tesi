'''
Shared helpers for every SAM-based analysis
'''

import numpy as np



def samples_per_class(dataloader, n_classes, n_per_class):
    # Collect n_per_class images for each class, in loader order
    samples = {c: [] for c in range(n_classes)}

    for images, labels in dataloader:
        for image, label in zip(images, labels):
            c = label.item()
            if len(samples[c]) < n_per_class:
                samples[c].append(image)

    return samples


def sam_stacks(net, image_list, time_steps, device, layer_index, gamma,
               get_layer_spikes, compute_sam):
    # Full SAM stack [T, H, W] per image
    stacks = []
    for image in image_list:
        spikes = get_layer_spikes(net, image, time_steps, device, layer_index)
        stacks.append(compute_sam(spikes, gamma).cpu())
    return stacks


# aggregation
def nanmean_or_nan(values):
    # Mean ignoring nan, returning nan when every entry is nan.
    if np.all(np.isnan(values)):
        return float('nan')
    return float(np.nanmean(values))



# Divergence metrics

def pearson(map_a, map_b):
    # Pearson correlation between two 2-D maps, flattened.
    a = map_a.flatten().numpy()
    b = map_b.flatten().numpy()

    if a.std() == 0 or b.std() == 0:
        return float('nan')

    return float(np.corrcoef(a, b)[0, 1])


def top_k_iou(map_a, map_b, k=0.20):
    # intersection-over-union of the top-k fraction of pixels in two maps
    a = map_a.flatten().numpy()
    b = map_b.flatten().numpy()

    mask_a = a > np.quantile(a, 1.0 - k)
    mask_b = b > np.quantile(b, 1.0 - k)

    # non-empty masks
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return float('nan')

    union = np.logical_or(mask_a, mask_b).sum()
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection / union)