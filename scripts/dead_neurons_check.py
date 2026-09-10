'''
Fraction of dead neurons, between risolutions
The firing rate increases monotonically in the 4th layer, escaping from
the 224px pathological 4th layer. But a greater mean firing rate can
derive 1) from more active neurons or 2) same neurons that spike more.
We now analyse this on validation set
'''

import argparse
import json
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import torch

from model.snn_model import GWGlitchSNN
from scripts.dataloader import build_dataloaders
from scripts.training_utils import direct_encode

RUNS_DIR = Path('runs')
DEAD_THRESHOLD = 0.01       
SATURATED_THRESHOLD = 0.80  
LAYERS = (1, 2, 3, 4)


def per_neuron_firing_rates(net, dataloader, time_steps, device):
    # mean firing rate medio of each neuron, per layer

    net.eval()
    totals = {l: None for l in LAYERS}
    n_seen = 0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            spikes = dict(zip(LAYERS, net(direct_encode(images, time_steps))[1:]))
            bs = images.shape[0]
            n_seen += bs
            for l in LAYERS:
                batch_mean = spikes[l].mean(dim=(0, 1)) * bs   # [C, H, W]
                totals[l] = batch_mean if totals[l] is None else totals[l] + batch_mean

    return {l: (totals[l] / n_seen).flatten().cpu() for l in LAYERS}


def analyse_run(run_json, device):
    with open(run_json) as f:
        summary = json.load(f)

    input_size = summary['input_size']
    hp = summary['hyperparameters']

    _, _, val_dl, _ = build_dataloaders(input_size=input_size, verbose=False)

    checkpoint = Path(run_json).parent / 'best_model.pt'
    net = GWGlitchSNN(beta=hp['beta'], input_size=input_size).to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device))

    rates = per_neuron_firing_rates(net, val_dl, hp['time_steps'], device)

    out = {'tag': summary['tag'], 'input_size': input_size, 'seed': summary['seed'],
           'beta': hp['beta'], 'time_steps': hp['time_steps'], 'layers': {}}
    for l in LAYERS:
        r = rates[l]
        out['layers'][l] = {
            'n_neurons': r.numel(),
            'pct_dead': float((r < DEAD_THRESHOLD).float().mean() * 100),
            'pct_saturated': float((r > SATURATED_THRESHOLD).float().mean() * 100),
            'mean_rate': float(r.mean()),
            'median_rate': float(r.median()),
        }

    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='*', default=None,
                        help='folders in runs/ to analyse')
    parser.add_argument('--out', default='runs/dead_neurons.json')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.runs:
        run_jsons = [RUNS_DIR / r / 'run.json' for r in args.runs]
    else:
        run_jsons = sorted(RUNS_DIR.glob('fp32_*/run.json'))

    results = []
    for rj in run_jsons:
        if not rj.exists():
            print(f'absent: {rj}')
            continue
        print(f'analysing {rj.parent.name}...')
        results.append(analyse_run(rj, device))

    # Group by resolution

    by_res = defaultdict(list)
    for r in results:
        by_res[r['input_size'], r['beta'], r['time_steps']].append(r)

    def label(k):
        size, beta, T = k
        return f'{size}px b{beta}, T{T}'

    order = sorted(by_res, key=lambda k: (-k[0], k[1]))

    def cell(rs, l, key):
        v = [r['layers'][l][key] for r in rs]
        return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print('\nDead nurons (firing rate < 0.01), mean +- dev.std. on seed')
    print(f'{"":<20}' + ''.join(f'{"layer"+str(l):>16}' for l in LAYERS) + f'{"n neurons L4":>16}')

    for k in order:
        rs = by_res[k]
        line = f'{label(k):<20}'
        for l in LAYERS:
            m, d = cell(rs, l , 'pct_dead')
            line += f'{m:>11.1f}+-{d:<4.1f}'
        line += f'{rs[0]["layers"][4]["n_neurons"]:>16}'
        print(line)

    print('\nMedian of firing rate per neuron')
    print(f'{"":<20}' + ''.join(f'{"layer"+str(l):>16}' for l in LAYERS))
    for k in order:
        rs = by_res[k]
        line = f'{label(k):<20}'
        for l in LAYERS:
            m, _ = cell(rs, l, 'median_rate')
            line += f'{m:>16.4f}'
        print(line)

    print('\nSaturated neuons (firing rate > 0.80)')
    print(f'{"":<20}' + ''.join(f'{"layer"+str(l):>16}' for l in LAYERS))
    for k in order:
        rs = by_res[k]
        line = f'{label(k):<20}'
        for l in LAYERS:
            m, _ = cell(rs, l, 'pct_saturated')
            line += f'{m:>16.1f}'
        print(line)

    print(f'\nResults in {args.out}')


if __name__ == '__main__':
    main()
