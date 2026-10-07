'''
Energy accounting: synaptic operations, memory and fine-tuning costs

'''

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn as nn

from model.snn_model import GWGlitchSNN
from scripts.training_utils import direct_encode
from xai.sam import MANIFEST

OUT_JSON = Path('results/energy.json')
RUNS_DIR = Path('runs')

# Horowitz 45 nm
FP32_MAC_PJ = 4.6
FP32_AC_PJ = 0.9
INT8_MAC_PJ = 0.2
INT8_AC_PJ = 0.03

# Which hidden layer's firing rate gates which weight layer
GATED_BY = {'conv2': 'layer1', 'conv3': 'layer2', 'conv4': 'layer3',
            'linear': 'layer4'}
ANALOG_INPUT_LAYER = 'conv1'


def ac_energy_pj(bits):
    # Accumulate energy- exact at 32 and 8 bits
    if bits is None or bits >= 32:
        return FP32_AC_PJ, False
    if bits == 8:
        return INT8_AC_PJ, False
    return INT8_AC_PJ * (bits / 8), True


def mac_energy_pj(bits):
    # MAC Multiply-accumulate energy - exact at 32 and 8 bits
    if bits is None or bits >= 32:
        return FP32_MAC_PJ, False
    if bits == 8:
        return INT8_MAC_PJ, False
    return INT8_MAC_PJ * (bits / 8) ** 2, True

# architecture

def build_reference_net(device):
    with open(MANIFEST, encoding='utf-8') as f:
        manifest = json.load(f)
    net = GWGlitchSNN(beta=manifest['hyperparameters']['beta'],
                      input_size=manifest['input_size']).to(device)
    net.eval()
    return net, manifest


def layer_macs(device=None):
    # MAC count and parameter count of each Conv2d and linear
    device = device or torch.device('cpu')
    net, manifest = build_reference_net(device)
    input_size = manifest['input_size']
    time_steps = manifest['hyperparameters']['time_steps']

    macs = {}
    params = {}
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            if isinstance(module, nn.Conv2d):
                out_elements = output.shape[1] * output.shape[2] * output.shape[3]
                per_output = (module.in_channels // module.groups
                              * module.kernel_size[0] * module.kernel_size[1])
                macs[name] = out_elements * per_output
            else:
                macs[name] = module.in_features * module.out_features
            params[name] = module.weight.numel()
        return hook

    for name, module in net.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(make_hook(name)))

    dummy = torch.zeros(1, 3, input_size, input_size, device=device)
    with torch.no_grad():
        net(direct_encode(dummy, time_steps=time_steps))

    for handle in handles:
        handle.remove()

    return macs, params, input_size, time_steps


def neuron_counts(device=None):
    #neurons per hidden LIF layer for the membrane state footprint

    device = device or torch.device('cpu')
    net, manifest = build_reference_net(device)
    dummy = torch.zeros(1, 3, manifest['input_size'], manifest['input_size'],
                        device=device)
    with torch.no_grad():
        _, *hidden = net(direct_encode(
            dummy, time_steps=manifest['hyperparameters']['time_steps']))
    return {f'layer{i}': int(s.shape[2:].numel())
            for i, s in enumerate(hidden, start=1)}

# accounting

def account(rates, macs, params, neurons, time_steps, weight_bits, membrane_bits):
    # SOPs, energy and memory for one configuration, per layer
    ac_pj, ac_extrapolated = ac_energy_pj(weight_bits)
    mac_pj, mac_extrapolated = mac_energy_pj(weight_bits)

    per_layer = {
        # conv1: analog input one forward for the whole inference
        ANALOG_INPUT_LAYER: {
            'operations': macs[ANALOG_INPUT_LAYER],
            'kind': 'MAC',
            'gated_by': None,
            'energy_pj': macs[ANALOG_INPUT_LAYER] * mac_pj,
        }
    }

    for name, rate_key in GATED_BY.items():
        sops = rates[rate_key] * time_steps * macs[name]
        per_layer[name] = {
            'operations': sops,
            'kind': 'AC',
            'gated_by': rate_key,
            'firing_rate': rates[rate_key],
            'energy_pj': sops * ac_pj,
        }

    weight_bytes = sum(params[n] * (weight_bits or 32) / 8 for n in params)
    membrane_bytes = sum(neurons[k] * (membrane_bits or 32) / 8 for k in neurons)

    return {
        'per_layer': per_layer,
        'total_sops': sum(v['operations'] for v in per_layer.values()
                          if v['kind'] == 'AC'),
        'total_macs': sum(v['operations'] for v in per_layer.values()
                          if v['kind'] == 'MAC'),
        'total_energy_pj': sum(v['energy_pj'] for v in per_layer.values()),
        'weight_bytes': weight_bytes,
        'membrane_bytes': membrane_bytes,
        'ac_pj': ac_pj,
        'mac_pj': mac_pj,
        'extrapolated': ac_extrapolated or mac_extrapolated,
    }


def read_runs(rate_field, input_size):
    # every run directory with a run.json, FP32 references included, restricted to the manifest resolution
    keep = re.compile(rf'^(fp32|qat_\w+)_{input_size}px_seed\d+$')
    runs = {}
    for path in sorted(RUNS_DIR.glob('*/run.json')):
        if not keep.match(path.parent.name):
            continue
        with open(path, encoding='utf-8') as f:
            summary = json.load(f)
        rates, source = summary.get(rate_field), rate_field
        if not rates:
            rates, source = summary.get('checkpoint_firing_rates'), 'checkpoint'
        if not rates:
            continue
        quantisation = summary.get('quantisation') or {}
        runs[path.parent.name] = {
            'arm': summary.get('arm', 'fp32-reference'),
            'weight_bits': quantisation.get('weight_bits'),
            'membrane_bits': quantisation.get('membrane_bits'),
            'seed': summary.get('seed'),
            'accuracy': summary.get('best_val_accuracy'),
            'epochs_run': summary.get('epochs_run'),
            'rates': rates,
            'rate_source': source,
        }
    return runs


def read_evaluated(input_size):
    # Rates measured by evaluate_model.py, one validation pass per model

    rates_dir = Path('results/grid')
    runs = {}
    for path in sorted(rates_dir.glob('*.json')):
        with open(path, encoding='utf-8') as f:
            record = json.load(f)
        if 'mean_rate' not in record:
            continue
        runs[record.get('tag', path.stem)] = {
            'arm': record.get('arm'),
            'weight_bits': record.get('weight_bits'),
            'membrane_bits': record.get('membrane_bits'),
            'seed': record.get('seed'),
            'accuracy': record.get('accuracy'),
            'epochs_run': None,
            # evaluate_model keys layers by integer- JSON turns them into strings
            'rates': {f'layer{k}': v for k, v in record['mean_rate'].items()},
            'rate_source': 'evaluated',
        }
    return runs


def baseline_for(tag, run, results, input_size, override=None):
    # the FP32 run of the same seed
    if override:
        return override.format(seed=run.get('seed'), input_size=input_size)
    if tag.startswith('fp32_'):
        return tag
    seed = run.get('seed')
    if seed is None:
        return None
    expected = f'fp32_{input_size}px_seed{seed}'
    return expected if expected in results else None


def main():
    parser = argparse.ArgumentParser(
        description='SOP, energy and memory accounting from the grid run.json')
    parser.add_argument('--rates', choices=['evaluated', 'checkpoint', 'final'],
                        default='evaluated',
                        help='evaluated: measured by evaluate_model.py. checkpoint: deployed model.'
                             'final: the model as fine-tuning left it')
    parser.add_argument('--baseline', type=str, default=None,
                        help='force one run as denominator; default: the FP32 '
                             'run of each run\'s own seed')
    parser.add_argument('--out', type=str, default=str(OUT_JSON),
                        help='where to write the result')
    args = parser.parse_args()
    out_json = Path(args.out)

    macs, params, input_size, time_steps = layer_macs()
    neurons = neuron_counts()

    if args.rates == 'evaluated':
        runs, rate_field = read_evaluated(input_size), 'evaluated'
    else:
        rate_field = ('final_firing_rates' if args.rates == 'final'
                      else 'checkpoint_firing_rates')
        runs = read_runs(rate_field, input_size)
    if not runs:
        raise SystemExit(f'no rates found for --rates {args.rates}')

    print(f'{input_size} px | T={time_steps} | rates from {rate_field}')
    print('\nMAC count per forward, parameters, and what gates each layer:')
    for name in macs:
        gate = GATED_BY.get(name, 'analog input, computed once')
        print(f'  {name:<8}{macs[name]:>14,} MAC{params[name]:>12,} params'
              f'   {gate}')
    print('  hidden neurons: ' + ', '.join(f'{k} {v:,}' for k, v in neurons.items()))

    results = {}
    for tag, run in runs.items():
        results[tag] = {**run, **account(run['rates'], macs, params, neurons,
                                         time_steps, run['weight_bits'],
                                         run['membrane_bits'])}

    bases = {t: baseline_for(t, r, results, input_size, args.baseline)
             for t, r in results.items()}
    orphans = [t for t, b in bases.items() if b is None or b not in results]
    if orphans:
        raise SystemExit(
            'no FP32 run of the same seed for: ' + ', '.join(sorted(orphans)))

    # report
    print('\nbaseline: the FP32 run of each seed' if not args.baseline
          else f'\nbaseline: {args.baseline} (forced)')
    header = (f'{"config":<28}{"SOP":>14}{"vs seed":>10}{"energy pJ":>13}'
              f'{"vs seed":>11}{"weights kB":>12}{"state kB":>10}{"rates":>11}')
    print('\n' + header)
    print('-' * len(header))
    for tag, r in results.items():
        base = results[bases[tag]]
        mark = '*' if r['extrapolated'] else ' '
        print(f'{tag:<28}{r["total_sops"]:>14,.0f}'
              f'{r["total_sops"] / base["total_sops"]:>9.3f}x'
              f'{r["total_energy_pj"]:>13,.0f}'
              f'{r["total_energy_pj"] / base["total_energy_pj"]:>10.3f}x{mark}'
              f'{r["weight_bytes"] / 1024:>12,.1f}'
              f'{r["membrane_bytes"] / 1024:>10,.1f}'
              f'{r["rate_source"]:>11}')

    if any(r['extrapolated'] for r in results.values()):
        print('\n* energy extrapolated from the INT8 anchor')

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'input_size': input_size, 'time_steps': time_steps,
                   'rate_field': rate_field, 'baselines': bases,
                   'macs': macs, 'params': params, 'neurons': neurons,
                   'horowitz_pj': {'fp32_mac': FP32_MAC_PJ, 'fp32_ac': FP32_AC_PJ,
                                   'int8_mac': INT8_MAC_PJ, 'int8_ac': INT8_AC_PJ},
                   'results': results}, f, indent=2)
    print(f'\nwritten to {out_json}')


if __name__ == '__main__':
    main()