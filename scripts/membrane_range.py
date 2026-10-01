'''
Measure the dynamic range of the membrane potential U[t], per layer.
For the membrane-only arm of the quantization grid
'''

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.dataloader import build_dataloaders
from scripts.training_utils import direct_encode
from xai.sam import MANIFEST, load_reference_model

OUT_JSON = Path('config/membrane.json')
N_BINS = 2000
THRESHOLD = 1.0 # snnTorch leaky default
PERCENTILES = (0.01, 0.1, 1.0, 50.0, 99.0, 99.9, 99.99)
BIT_WIDTHS = (8, 4, 2)

LIF_NAMES = ('lif1', 'lif2', 'lif3', 'lif4', 'lif_out') # redout neuron carries a membrane too, even if not part of SAM

class MembraneCapture:
    '''
    Forward hooks that expose the membrane potential
    snn.Leaky.forward returns the tuple (spk, mem) when init_hidden is False. The hook reads output[1]
    '''
    def __init__(self, net, names=LIF_NAMES):
        self.names = names
        self.handles = []
        self.callback = None

        for name in names:
            module = getattr(net, name)
            self.handles.append(
                module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, inputs, output):
            if self.callback is None:
                return
            # output is (spk, mem); mem is post-update, post-reset
            mem = output[1] if isinstance(output, tuple) else output
            self.callback(name, mem.detach())
        return hook
 
    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
 
# extremes

def measure_extremes(net, capture, dataloader, time_steps, device):
    # min and max per layer over the whole partition
    lo = {name: float('inf') for name in capture.names}
    hi = {name: float('-inf') for name in capture.names}
 
    def on_membrane(name, mem):
        lo[name] = min(lo[name], mem.min().item())
        hi[name] = max(hi[name], mem.max().item())
 
    capture.callback = on_membrane
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            net(direct_encode(images.to(device), time_steps))
            if (i + 1) % 15 == 0:
                print(f'    pass 1: batch {i + 1}')
    capture.callback = None
 
    return lo, hi
 
 
# distribution

def measure_distribution(net, capture, dataloader, time_steps, device, lo, hi):
    # histogram, mean, variance and above-threshold fraction per layer

    edges = {n: np.linspace(lo[n], hi[n], N_BINS + 1) for n in capture.names}
    counts = {n: np.zeros(N_BINS, dtype=np.int64) for n in capture.names}
    total = defaultdict(int)
    sum_x = defaultdict(float)
    sum_x2 = defaultdict(float)
    above = defaultdict(int)
 
    def on_membrane(name, mem):
        flat = mem.flatten().cpu().numpy()
        counts[name] += np.histogram(flat, bins=edges[name])[0]
        total[name] += flat.size
        sum_x[name] += float(flat.sum())
        sum_x2[name] += float((flat.astype(np.float64) ** 2).sum())
        above[name] += int((flat > THRESHOLD).sum())
 
    capture.callback = on_membrane
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            net(direct_encode(images.to(device), time_steps))
            if (i + 1) % 15 == 0:
                print(f'    pass 2: batch {i + 1}')
    capture.callback = None
 
    return edges, counts, dict(total), dict(sum_x), dict(sum_x2), dict(above)
 
 
def percentile_from_histogram(counts, edges, q):
    # Read a percentile back from a histogram, at bin-centre resolution
    
    cumulative = np.cumsum(counts)
    target = q / 100.0 * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, target)), len(counts) - 1)
    return float(edges[index] + (edges[index + 1] - edges[index]) * 0.5)
 

# WEIGHTS
def analyse_weights(net):
    '''
    Per-tensor against per-channel quantisation, as a number
 
    below 1 bit lost, per-tensor is fine and simpler to implement
    on hardware. above 2, per-channel is worth the extra scale factors
    '''
    report = {}
    for name, module in net.named_modules():
    
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            continue
 
        weight = module.weight.detach()
        per_tensor_max = weight.abs().max().item()
        # one scale per output channel
        per_channel_max = weight.abs().flatten(1).max(dim=1).values
 
        smallest = per_channel_max.min().item()
        ratio = per_tensor_max / smallest if smallest > 0 else float('inf')
 
        report[name] = {
            'shape': list(weight.shape),
            'per_tensor_max_abs': per_tensor_max,
            'per_channel_max_abs_min': smallest,
            'per_channel_max_abs_max': per_channel_max.max().item(),
            'ratio_worst_channel': ratio,
            'effective_bits_lost': float(np.log2(ratio)) if ratio != float('inf') else None,
            'std': weight.std().item(),
        }
    return report

 
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, time_steps, input_size, layer_index, gamma = load_reference_model(
        device, MANIFEST)
    print(f'{input_size} px | T={time_steps} | layer {layer_index} | gamma {gamma}')
    print(f'firing threshold assumed at {THRESHOLD}\n')
 
    _, _, val_dataloader, _ = build_dataloaders(input_size=input_size,
                                                verbose=False)
    n_images = len(val_dataloader.dataset)
    print(f'{n_images} validation images, two passes\n')
 
    capture = MembraneCapture(net)
    try:
        print('  measuring extremes...')
        lo, hi = measure_extremes(net, capture, val_dataloader, time_steps, device)
 
        print('  measuring distribution...')
        edges, counts, total, sum_x, sum_x2, above = measure_distribution(
            net, capture, val_dataloader, time_steps, device, lo, hi)
    finally:
        capture.remove()

    layers = {}
    for name in LIF_NAMES:
        n = total[name]
        mean = sum_x[name] / n
        variance = max(0.0, sum_x2[name] / n - mean ** 2)
        percentiles = {str(q): percentile_from_histogram(counts[name],
                                                         edges[name], q)
                       for q in PERCENTILES}
        layers[name] = {
            'min': lo[name],
            'max': hi[name],
            'mean': mean,
            'std': variance ** 0.5,
            'percentiles': percentiles,
            'n_values': n,
            'fraction_above_threshold': above[name] / n,
            'histogram': {'edges': edges[name].tolist(),
                          'counts': counts[name].tolist()},
        }
 
    weights = analyse_weights(net)
 
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'manifest': str(MANIFEST),
                   'input_size': input_size,
                   'time_steps': time_steps,
                   'threshold': THRESHOLD,
                   'n_images': n_images,
                   'n_bins': N_BINS,
                   'membrane': layers,
                   'weights': weights}, f, indent=2)
 
    # report
    print(f'\nMEMBRANE POTENTIAL, per layer')
    header = (f'{"layer":<10}{"min":>9}{"max":>9}{"mean":>9}{"std":>9}'
              f'{"p0.1":>9}{"p99.9":>9}{"> thr":>9}')
    print(header)
    print('-' * len(header))
    for name in LIF_NAMES:
        s = layers[name]
        print(f'{name:<10}{s["min"]:>9.3f}{s["max"]:>9.3f}{s["mean"]:>9.3f}'
              f'{s["std"]:>9.3f}{s["percentiles"]["0.1"]:>9.3f}'
              f'{s["percentiles"]["99.9"]:>9.3f}'
              f'{100 * s["fraction_above_threshold"]:>8.2f}%')
 
    
    print(f'\nQUANTISATION STEP AS A FRACTION OF THE THRESHOLD ({THRESHOLD})')
    
 
    for criterion, key_lo, key_hi in (('absolute min/max', None, None),
                                      ('0.1 - 99.9 percentile', '0.1', '99.9'),
                                      ('1 - 99 percentile', '1.0', '99.0')):
        print(f'  {criterion}')
        print(f'    {"layer":<10}{"range":>10}' +
              ''.join(f'{str(b) + " bit":>12}' for b in BIT_WIDTHS) +
              f'{"clipped":>10}')
        for name in LIF_NAMES:
            s = layers[name]
            if key_lo is None:
                span = s['max'] - s['min']
                clipped = 0.0
            else:
                span = s['percentiles'][key_hi] - s['percentiles'][key_lo]
                clipped = 2 * float(key_lo)
            row = f'    {name:<10}{span:>10.3f}'
            for b in BIT_WIDTHS:
                step = span / (2 ** b - 1)
                row += f'{100 * step / THRESHOLD:>11.1f}%'
            row += f'{clipped:>9.2f}%'
            print(row)
        print()
 
    # weights
    print('WEIGHTS: per-tensor against per-channel')
    header = (f'{"layer":<10}{"shape":>20}{"max|w|":>10}{"worst ch":>10}'
              f'{"ratio":>8}{"bits lost":>11}')
    print(header)
    print('-' * len(header))
    for name, w in weights.items():
        print(f'{name:<10}{str(w["shape"]):>20}{w["per_tensor_max_abs"]:>10.4f}'
              f'{w["per_channel_max_abs_min"]:>10.4f}{w["ratio_worst_channel"]:>8.1f}'
              f'{w["effective_bits_lost"]:>11.2f}')
 
    worst = max(w['effective_bits_lost'] for w in weights.values())
    print(f'\n  worst channel loses {worst:.2f} bits under per-tensor quantisation')
    if worst < 1.0:
        print('  -> ok per-tensor, and simpler on hardware')
    elif worst < 2.0:
        print('  -> borderline: per-tensor at 8 bit, per-channel at 4 and 2')
    else:
        print('  -> per-channel worth its extra scale factors')
 
    print(f'\nwritten to {OUT_JSON}')
 
 
if __name__ == '__main__':
    main()