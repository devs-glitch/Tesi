'''
Per-image CV of the inter-spike intervals, measured inside a single image
'''

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.dataloader import build_dataloaders
from scripts.evaluate_model import load_model
from scripts.training_utils import direct_encode

MANIFEST = Path('config/baseline_manifest.json')
OUT = Path('results/per_image_cv.json')
LAYERS = (1, 2, 3, 4)

# The six models carrying a 2-bit membrane, and the three parents they are read against
DEFAULT_MODELS = (
    'fp32_112px_seed1234', 'fp32_112px_seed2345', 'fp32_112px_seed3456',
    'qat_wxu2_112px_seed1234', 'qat_wxu2_112px_seed2345', 'qat_wxu2_112px_seed3456',
    'qat_w2u2_112px_seed1234', 'qat_w2u2_112px_seed2345', 'qat_w2u2_112px_seed3456',
)


def pattern_table(time_steps):
    # n_spikes and within-image CV for every one of the 2^T spike patterns
    size = 2 ** time_steps
    n_spikes = np.zeros(size, dtype=np.int64)
    cv = np.full(size, np.nan)
    for code in range(size):
        times = np.flatnonzero([(code >> t) & 1 for t in range(time_steps)])
        n_spikes[code] = times.size
        if times.size >= 2:
            isi = np.diff(times).astype(float)
            cv[code] = float(isi.std() / isi.mean())
    return n_spikes, cv


def pattern_histogram(net, dataloader, time_steps, device):
    # Count, per layer, how often each spike pattern occurs over the partition
    weights = (2 ** torch.arange(time_steps, device=device)).view(-1, 1, 1)
    counts = {l: np.zeros(2 ** time_steps, dtype=np.int64) for l in LAYERS}

    net.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _, *hidden = net(direct_encode(images, time_steps))
            for layer, spikes in zip(LAYERS, hidden):
                # [T, B, C, H, W] -> [T, B, N] -> one integer code per (image, neuron)
                flat = spikes.reshape(spikes.shape[0], spikes.shape[1], -1)
                codes = (flat.to(torch.int64) * weights).sum(dim=0)
                counts[layer] += np.bincount(
                    codes.reshape(-1).cpu().numpy(),
                    minlength=2 ** time_steps).astype(np.int64)
    return counts


def summarise(counts, n_spikes, cv):
    out = {}
    for layer, hist in counts.items():
        total = int(hist.sum())
        one_interval = n_spikes >= 2      # a CV exists, but may be trivially zero
        two_intervals = n_spikes >= 3     # a CV can distinguish regular from not

        n1, n2 = int(hist[one_interval].sum()), int(hist[two_intervals].sum())
        zero2 = int(hist[two_intervals & (cv == 0)].sum())

        def mean_cv(mask, n):
            return float((hist[mask] * cv[mask]).sum() / n) if n else float('nan')

        out[str(layer)] = {
            'n_neuron_images': total,
            'fraction_with_an_interval': n1 / total if total else float('nan'),
            'fraction_with_two_intervals': n2 / total if total else float('nan'),
            'mean_cv_from_one_interval': mean_cv(one_interval, n1),
            'mean_cv_from_two_intervals': mean_cv(two_intervals, n2),
            'fraction_cv_exactly_zero': zero2 / n2 if n2 else float('nan'),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='*', default=list(DEFAULT_MODELS))
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(MANIFEST, encoding='utf-8') as f:
        manifest = json.load(f)
    input_size = manifest['input_size']
    _, _, val_dataloader, _ = build_dataloaders(input_size=input_size,
                                                verbose=False)

    results = {}
    table_cache = {}
    for name in args.models:
        net, handle, time_steps, _, summary = load_model(name, device)
        if time_steps not in table_cache:
            table_cache[time_steps] = pattern_table(time_steps)
        n_spikes, cv = table_cache[time_steps]

        counts = pattern_histogram(net, val_dataloader, time_steps, device)
        results[name] = {'arm': summary.get('arm'), 'seed': summary.get('seed'),
                         'time_steps': time_steps,
                         'layers': summarise(counts, n_spikes, cv)}
        if handle is not None:
            handle.remove()

        one = results[name]['layers']['1']
        print(f"{name:28s} layer 1 | "
              f"CV from >=2 intervals {one['mean_cv_from_two_intervals']:.4f} "
              f"(on {one['fraction_with_two_intervals'] * 100:5.2f} % of "
              f"neuron-images, {one['fraction_cv_exactly_zero'] * 100:5.2f} % "
              f"of them exactly zero) | "
              f"CV from >=1 interval {one['mean_cv_from_one_interval']:.4f} "
              f"(on {one['fraction_with_an_interval'] * 100:5.2f} %)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'\n{args.out}')


if __name__ == '__main__':
    main()