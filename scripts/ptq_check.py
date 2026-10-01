'''
Post-training quantisation check before training 

GATE: does joint 8-bit match FP32?
Does bake() survive a save and reload?
What does PTQ alone cost at 4 and 2 bits?
'''

import json
import tempfile
from pathlib import Path

import torch

from model.quantization import apply_quantisation, MEMBRANE_RANGE
from scripts.dataloader import build_dataloaders
from scripts.training_utils import direct_encode
from xai.sam import MANIFEST, load_reference_model

OUT_JSON = Path('results/ptq_check.json')

DEAD_THRESHOLD = 0.01

ACCURACY_TOLERANCE = 4 / 1435

# (label, weight_bits, membrane_bits)

CONFIGS = [
    ('fp32',          None, None),

    ('joint 8b',      8,    8),      # the gate

    ('W-only 8b',     8,    None),
    ('W-only 4b',     4,    None),
    ('W-only 2b',     2,    None),

    ('U-only 8b',     None, 8),
    ('U-only 4b',     None, 4),
    ('U-only 2b',     None, 2),

    ('joint 4b',      4,    4),
    ('joint 2b',      2,    2),
]


# Evaluation

def evaluate(net, dataloader, time_steps, device):
    # One validation pass: accuracy, per-neuron firing rate, spike timing

    net.eval()
    correct = total = 0
    per_neuron = {}
    weighted_time = {}
    spike_total = {}
    n_seen = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            spike_out, *hidden = net(direct_encode(images, time_steps))

            predictions = spike_out.sum(dim=0).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()

            n_seen += images.shape[0]
            for index, spikes in enumerate(hidden, start=1):
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


def predictions_on(net, batch, time_steps):
    # raw output spike counts for one fixed batch

    net.eval()
    with torch.no_grad():
        spike_out, *_ = net(direct_encode(batch, time_steps))
        return spike_out.sum(dim=0).cpu()


def check_bake_round_trip(device, batch, bits=4):
    # Quantise, bake, save, reload into a fresh plain model, compare
    net, time_steps, _, _, _ = load_reference_model(device, MANIFEST)

    handle = apply_quantisation(net, weight_bits=bits)
    before = predictions_on(net, batch, time_steps)

    handle.bake()
    after_bake = predictions_on(net, batch, time_steps)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'baked.pt'
        torch.save(net.state_dict(), path)

        fresh, _, _, _, _ = load_reference_model(device, MANIFEST)
        missing, unexpected = fresh.load_state_dict(
            torch.load(path, map_location=device), strict=False)
        after_reload = predictions_on(fresh, batch, time_steps)

    return {
        'bits': bits,
        'bake_drift': float((after_bake - before).abs().max()),
        'reload_drift': float((after_reload - before).abs().max()),
        'missing_keys': list(missing) if missing else [],
        'unexpected_keys': list(unexpected) if unexpected else [],
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, time_steps, input_size, _, _ = load_reference_model(device, MANIFEST)

    _, _, val_dataloader, _ = build_dataloaders(input_size=input_size,
                                                verbose=False)
    print(f'{input_size} px | T={time_steps} | '
          f'{len(val_dataloader.dataset)} validation images')
    print(f'membrane range {MEMBRANE_RANGE}, post-training - no fine-tuning\n')

    results = {}
    weight_steps = {}

    for label, w_bits, u_bits in CONFIGS:
        print(f'  {label}...')
        if w_bits is None and u_bits is None:
            results[label] = evaluate(net, val_dataloader, time_steps, device)
            continue
        with apply_quantisation(net, weight_bits=w_bits,
                                membrane_bits=u_bits) as handle:
            results[label] = evaluate(net, val_dataloader, time_steps, device)
            results[label]['config'] = handle.config()
            if w_bits is not None:
                weight_steps[label] = handle.weight_steps()

    # remove() must leave the network exactly as it was
    print('  fp32 (repeat, guards against state left behind)...')
    repeat = evaluate(net, val_dataloader, time_steps, device)
    restore_ok = repeat['accuracy'] == results['fp32']['accuracy']

    print('  bake() round-trip...')
    fixed_batch = next(iter(val_dataloader))[0].to(device)
    round_trip = check_bake_round_trip(device, fixed_batch)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'manifest': str(MANIFEST),
                   'membrane_range': list(MEMBRANE_RANGE),
                   'accuracy_tolerance': ACCURACY_TOLERANCE,
                   'mode': 'post-training quantisation, no fine-tuning',
                   'restore_exact': restore_ok,
                   'bake_round_trip': round_trip,
                   'weight_steps': weight_steps,
                   'results': results}, f, indent=2)

   #report
    baseline = results['fp32']

    print(f'\nAccuracy (FP32 {baseline["accuracy"]:.4f}, '
          f'tolerance {ACCURACY_TOLERANCE:.4f} = 4 samples)')
    header = f'{"config":<14}{"accuracy":>10}{"delta":>10}{"samples":>10}'
    print(header)
    print('-' * len(header))
    for label, r in results.items():
        delta = r['accuracy'] - baseline['accuracy']
        samples = delta * len(val_dataloader.dataset)
        print(f'{label:<14}{r["accuracy"]:>10.4f}{delta:>+10.4f}'
              + ('' if label == 'fp32' else f'{samples:>+10.0f}'))

    print('\nMean firing rate per layer')
    print(f'{"config":<14}' + ''.join(f'{"layer" + str(i):>12}' for i in (1, 2, 3, 4)))
    for label, r in results.items():
        print(f'{label:<14}' + ''.join(f'{r["mean_rate"][i]:>12.4f}' for i in (1, 2, 3, 4)))

    print('\nDead fraction per layer (rate < 0.01)')
    print(f'{"config":<14}' + ''.join(f'{"layer" + str(i):>12}' for i in (1, 2, 3, 4)))
    for label, r in results.items():
        print(f'{label:<14}' + ''.join(f'{100 * r["dead_fraction"][i]:>11.1f}%'
                                       for i in (1, 2, 3, 4)))

    print(f'\nSpike timing - temporal centre of mass, in steps (T = {time_steps})')
    print(f'{"config":<14}' + ''.join(f'{"layer" + str(i):>13}' for i in (1, 2, 3, 4)))
    for label, r in results.items():
        row = f'{label:<14}'
        for i in (1, 2, 3, 4):
            delta = r['spike_com'][i] - baseline['spike_com'][i]
            row += (f'{r["spike_com"][i]:>9.3f}    ' if label == 'fp32'
                    else f'{r["spike_com"][i]:>8.3f}{delta:>+5.3f}')
        print(row)

    if weight_steps:
        print('\nWeight quantization step, mean over channels')
        layers = list(next(iter(weight_steps.values())).keys())
        print(f'{"config":<14}' + ''.join(f'{n:>14}' for n in layers))
        for label, steps in weight_steps.items():
            print(f'{label:<14}' + ''.join(f'{steps[n]:>14.5f}' for n in layers))

    print('\n' + '-' * 60)

    gate = results['joint 8b']
    gate_delta = abs(gate['accuracy'] - baseline['accuracy'])
    gate_samples = gate_delta * len(val_dataloader.dataset)
    gate_shift = max(abs(gate['spike_com'][i] - baseline['spike_com'][i])
                     for i in (1, 2, 3, 4))

    print(f'GATE  joint 8-bit vs FP32: {gate_samples:.0f} samples '
          f'(limit 4), timing shift {gate_shift:.4f} steps')
    if gate_delta <= ACCURACY_TOLERANCE:
        print('      The quantiser is wired to the network correctly')
        print('      For scale, the SAM seed-to-seed floor on the temporal centre')
        print(f'      of mass is 0.115 steps; this is {100 * gate_shift / 0.115:.1f}% of it.')
    else:
        print('      Fail. at 8 bits the steps are far below anything that moved')

    print(f'\nRestore  removing the quantiser returns the FP32 network: '
          f'{"exact" if restore_ok else "NOT EXACT"}')
    if not restore_ok:
        print('      Every row below the first is contaminated by the one above')

    rt = round_trip
    rt_ok = (rt['bake_drift'] == 0.0 and rt['reload_drift'] == 0.0
             and not rt['missing_keys'] and not rt['unexpected_keys'])
    print(f'\nBake fold drift {rt["bake_drift"]:.3e}, '
          f'save-reload drift {rt["reload_drift"]:.3e} '
          f'({"exact" if rt_ok else "NOT EXACT"})')
    if not rt_ok:
        if rt['missing_keys'] or rt['unexpected_keys']:
            print(f'      missing keys: {rt["missing_keys"][:4]}')
            print(f'      unexpected keys: {rt["unexpected_keys"][:4]}')
        else:
            print('      The weights do not survive the round-trip unchanged')

    print('-' * 60)
    print(f'\nwritten to {OUT_JSON}')


if __name__ == '__main__':
    main()