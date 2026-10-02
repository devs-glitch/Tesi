'''
Energy accounting: synaptic operations, memory and fine-tuning costs.

'''

import argparse
import json
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


def read_runs(rate_field):
    # Every run directory with a run.json
    runs = {}
    for path in sorted(RUNS_DIR.glob('*/run.json')):
        with open(path, encoding='utf-8') as f:
            summary = json.load(f)
        rates = summary.get(rate_field) or summary.get('checkpoint_firing_rates')
        if not rates:
            continue
        runs[path.parent.name] = {
            'arm': summary.get('arm', 'fp32-reference'),
            'weight_bits': summary.get('quantisation', {}).get('weight_bits'),
            'membrane_bits': summary.get('quantisation', {}).get('membrane_bits'),
            'seed': summary.get('seed'),
            'accuracy': summary.get('best_val_accuracy'),
            'epochs_run': summary.get('epochs_run'),
            'rates': rates,
        }
    return runs

def main():
    parser = argparse.ArgumentParser(
        description='SOP, energy and memory accounting from the grid run.json files.')
    parser.add_argument('--rates', choices=['checkpoint', 'final'],
                        default='checkpoint',
                        help='checkpoint: the model that would be deployed. '
                             'final: the model as fine-tuning left it')
    parser.add_argument('--baseline', type=str, default=None,
                        help='run to express ratios against; default: the first '
                             'fp32_* run found')
    args = parser.parse_args()

    rate_field = ('final_firing_rates' if args.rates == 'final'
                  else 'checkpoint_firing_rates')
    runs = read_runs(rate_field)
    if not runs:
        raise SystemExit(f'no run.json with {rate_field} found under {RUNS_DIR}/')

    macs, params, input_size, time_steps = layer_macs()
    neurons = neuron_counts()

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

    baseline_tag = args.baseline or next(
        (t for t in results if t.startswith('fp32_')), None)
    if baseline_tag is None:
        raise SystemExit('no FP32 baseline found; pass --baseline')
    baseline = results[baseline_tag]

    #report
    print(f'\nbaseline: {baseline_tag}')
    header = (f'{"config":<28}{"SOP":>14}{"vs base":>10}{"energy pJ":>13}'
              f'{"vs base":>11}{"weights kB":>12}{"state kB":>10}')
    print('\n' + header)
    print('-' * len(header))
    for tag, r in results.items():
        mark = '*' if r['extrapolated'] else ' '
        print(f'{tag:<28}{r["total_sops"]:>14,.0f}'
              f'{r["total_sops"] / baseline["total_sops"]:>9.3f}x'
              f'{r["total_energy_pj"]:>13,.0f}'
              f'{r["total_energy_pj"] / baseline["total_energy_pj"]:>10.3f}x{mark}'
              f'{r["weight_bytes"] / 1024:>12,.1f}'
              f'{r["membrane_bytes"] / 1024:>10,.1f}')

    if any(r['extrapolated'] for r in results.values()):
        print('\n* energy extrapolated from the INT8 anchor: AC linear in bits,')
        print('  MAC quadratic. The SOP and memory columns are exact and do not')
        print('  depend on that assumption')

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'input_size': input_size, 'time_steps': time_steps,
                   'rate_field': rate_field, 'baseline': baseline_tag,
                   'macs': macs, 'params': params, 'neurons': neurons,
                   'horowitz_pj': {'fp32_mac': FP32_MAC_PJ, 'fp32_ac': FP32_AC_PJ,
                                   'int8_mac': INT8_MAC_PJ, 'int8_ac': INT8_AC_PJ},
                   'results': results}, f, indent=2)
    print(f'\nwritten to {OUT_JSON}')


if __name__ == '__main__':
    main()