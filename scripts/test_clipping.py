'''
Does clipping the membrane potential change the network's behaviour?
Clip network potential to see if behavior changes. Test before quantization

'''

import json
from pathlib import Path

import torch

from scripts.dataloader import build_dataloaders
from scripts.training_utils import direct_encode
from xai.sam import MANIFEST, load_reference_model

MEMBRANE_JSON = Path('config/membrane.json')
OUT_JSON = Path('results/clipping_test.json')

CLIP_LAYERS = ('lif1', 'lif2', 'lif3', 'lif4')

DEAD_THRESHOLD = 0.01

# Accuracy noise floor: the three-seed FP32 reference spans four samples out of 1435, so a difference smaller than this is not a difference
ACCURACY_TOLERANCE = 4 / 1435

# Clipping

class ClippingHooks:
    # clamp the membrane potential in place per layer

    def __init__(self, net, ranges):
        self.handles = []
        for name, (lo, hi) in ranges.items():
            module = getattr(net, name)
            self.handles.append(module.register_forward_hook(self._make(lo, hi)))

    @staticmethod
    def _make(lo, hi):
        def hook(module, inputs, output):
            spk, mem = output
            return spk, mem.clamp(lo, hi)
        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


# candidate ranges

def from_percentiles(membrane, low_key, high_key):
    # Per-layer range read from the measured distribution
    return {name: (membrane[name]['percentiles'][low_key],
                   membrane[name]['percentiles'][high_key])
            for name in CLIP_LAYERS}


def fixed(low, high):
    # the same range for every layer for comparison
    return {name: (low, high) for name in CLIP_LAYERS}

def quantisation_step(lo, hi, bits):
    '''
    step the quantiser would actually use for this range

    The scheme depends on the sign of the range:

      lo >= 0   unsigned, 2^b levels over [0, hi]:  step = hi / (2^b - 1)
      lo <  0   symmetric signed over r = max(|lo|, hi), zero exactly representable: step = r / (2^(b-1) - 1)
    '''

    if lo >= 0:
        return hi / (2 ** bits - 1)
    return max(abs(lo), abs(hi)) / (2 ** (bits - 1) - 1)

def build_candidates(membrane):
    # Ranges to test, from widest to narrowest

    return {
        'none': None,
        'p0.1-p99.9': from_percentiles(membrane, '0.1', '99.9'),
        'p1-p99': from_percentiles(membrane, '1.0', '99.0'),
        'fixed [-8, +4]': fixed(-8.0, 4.0),
        'fixed [-4, +4]': fixed(-4.0, 4.0),
        'fixed [-2, +2]': fixed(-2.0, 2.0),
        'sanity [-0.5, +0.5]': fixed(-0.5, 0.5),
        'unsigned [0, +4]': fixed(0.0, 4.0),
        'unsigned [0, +2]': fixed(0.0, 2.0),
    }

# Evaluation

def evaluate(net, dataloader, time_steps, device):
    """
    Accuracy, per-neuron firing rate and spike timing

    The firing rate is accumulated per neuron

    The temporal centre of mass, sum_t t*S_t / sum_t S_t catches timing: if the
    clamp lets neurons reach threshold sooner, the centre of mass moves earlier
    """
    net.eval()
    correct = total = 0
    per_neuron = {}
    weighted_time = {}       # sum over t of t * spikes
    spike_total = {}         # sum over t of spikes
    n_seen = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            spike_out, *hidden = net(direct_encode(images, time_steps))

            predictions = spike_out.sum(dim=0).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()

            batch = images.shape[0]
            n_seen += batch
            for index, spikes in enumerate(hidden, start=1):
                # spikes: [T, B, C, H, W] -> mean over time, sum over batch
                summed = spikes.mean(dim=0).sum(dim=0)
                if index not in per_neuron:
                    per_neuron[index] = torch.zeros_like(summed)
                    weighted_time[index] = 0.0
                    spike_total[index] = 0.0
                per_neuron[index] += summed

                steps = torch.arange(spikes.shape[0], dtype=spikes.dtype,
                                     device=spikes.device).view(-1, 1, 1, 1, 1)
                weighted_time[index] += float((steps * spikes).sum())
                spike_total[index] += float(spikes.sum())

    rates = {i: (v / n_seen).flatten().cpu() for i, v in per_neuron.items()}
    return {
        'accuracy': correct / total,
        'mean_rate': {i: float(r.mean()) for i, r in rates.items()},
        'dead_fraction': {i: float((r < DEAD_THRESHOLD).float().mean())
                          for i, r in rates.items()},
        'spike_com': {i: (weighted_time[i] / spike_total[i]
                          if spike_total[i] > 0 else float('nan'))
                      for i in weighted_time},
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, time_steps, input_size, _, _ = load_reference_model(device, MANIFEST)

    with open(MEMBRANE_JSON, encoding='utf-8') as f:
        membrane = json.load(f)['membrane']

    _, _, val_dataloader, _ = build_dataloaders(input_size=input_size,
                                                verbose=False)
    print(f'{input_size} px | T={time_steps} | '
          f'{len(val_dataloader.dataset)} validation images')
    print(f'clipping {", ".join(CLIP_LAYERS)}\n')

    candidates = build_candidates(membrane)
    results = {}

    for label, ranges in candidates.items():
        print(f'  {label}...')
        if ranges is None:
            results[label] = evaluate(net, val_dataloader, time_steps, device)
        else:
            with ClippingHooks(net, ranges):
                results[label] = evaluate(net, val_dataloader, time_steps, device)
            results[label]['ranges'] = {n: list(r) for n, r in ranges.items()}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'manifest': str(MANIFEST),
                   'clip_layers': list(CLIP_LAYERS),
                   'accuracy_tolerance': ACCURACY_TOLERANCE,
                   'results': results}, f, indent=2)

    # Report
    baseline = results['none']
    print(f'\nAccuracy (baseline {baseline["accuracy"]:.4f}, '
          f'tolerance {ACCURACY_TOLERANCE:.4f} = 4 samples)')
    header = f'{"range":<16}{"accuracy":>10}{"delta":>10}{"within":>9}'
    print(header)
    print('-' * len(header))
    for label, r in results.items():
        delta = r['accuracy'] - baseline['accuracy']
        within = '-' if label == 'none' else ('yes' if abs(delta) <= ACCURACY_TOLERANCE else 'NO')
        print(f'{label:<16}{r["accuracy"]:>10.4f}{delta:>+10.4f}{within:>9}')

    print('\nMean firing rate per layer')
    print(f'{"range":<16}' + ''.join(f'{"layer" + str(i):>12}' for i in (1, 2, 3, 4)))
    for label, r in results.items():
        print(f'{label:<16}' + ''.join(f'{r["mean_rate"][i]:>12.4f}' for i in (1, 2, 3, 4)))

    print('\nDead fraction per layer (rate < 0.01)')
    print(f'{"range":<16}' + ''.join(f'{"layer" + str(i):>12}' for i in (1, 2, 3, 4)))
    for label, r in results.items():
        print(f'{label:<16}' + ''.join(f'{100 * r["dead_fraction"][i]:>11.1f}%'
                                       for i in (1, 2, 3, 4)))

    # the explanation is built from when the spikes happen

    print(f'\nSpike timing - temporal centre of mass, in steps (T = {time_steps})')
    print(f'{"range":<16}' + ''.join(f'{"layer" + str(i):>12}' for i in (1, 2, 3, 4)))
    for label, r in results.items():
        row = f'{label:<16}'
        for i in (1, 2, 3, 4):
            delta = r['spike_com'][i] - baseline['spike_com'][i]
            row += (f'{r["spike_com"][i]:>8.3f}     ' if label == 'none'
                    else f'{r["spike_com"][i]:>8.3f}{delta:>+5.3f}')
        print(row)

    print('\nQuantization step implied by each range, as % of threshold')
    print(f'{"range":<18}' + ''.join(f'{str(b) + " bit":>12}' for b in (8, 4, 2)))
    for label, r in results.items():
        if 'ranges' not in r:
            continue
        lo, hi = max(r['ranges'].values(), key=lambda v: max(abs(v[0]), abs(v[1])))
        steps = [quantisation_step(lo, hi, b) for b in (8, 4, 2)]
        print(f'{label:<18}' + ''.join(f'{100 * s:>11.1f}%' for s in steps))

    print('\nRecomendation')
    acceptable = []
    for label, r in results.items():
        if label == 'none':
            continue
        if abs(r['accuracy'] - baseline['accuracy']) > ACCURACY_TOLERANCE:
            continue
        # check firing rate to see is a clamp changes dead neurons frac even at constant accuracy
        drift = max(abs(r['mean_rate'][i] - baseline['mean_rate'][i])
                    / baseline['mean_rate'][i] for i in (1, 2, 3, 4))
        shift = max(abs(r['spike_com'][i] - baseline['spike_com'][i])
                    for i in (1, 2, 3, 4))
        acceptable.append((label, drift, shift, r))

    if not acceptable:
        print('  quantiser must carry the full range: which means the 2-bit')
    else:
        print(f'  {"range":<16}{"widest span":>13}{"rate drift":>12}{"timing shift":>14}')
        for label, drift, shift, r in acceptable:
            spans = [hi - lo for lo, hi in
                     (tuple(v) for v in r['ranges'].values())]
            print(f'  {label:<16}{max(spans):>13.2f}{100 * drift:>11.1f}%'
                  f'{shift:>14.3f}')

        narrowest = min(acceptable,
                        key=lambda x: max(hi - lo for lo, hi in
                                          (tuple(v) for v in x[3]['ranges'].values())))
        print(f'\n  Narrowest range that holds accuracy: {narrowest[0]}')

    print(f'\nwritten to {OUT_JSON}')


if __name__ == '__main__':
    main()