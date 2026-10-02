'''
Runs the quantisation grid: every arm, every bit-width, every seed
'''

import argparse
import json
import re
import time
import traceback
from datetime import timedelta
from pathlib import Path

import torch

from scripts.train import load_hyperparams, RUNS_DIR, INPUT_SIZE
from scripts.train_quantised import (run_qat, config_tag, fp32_checkpoint_for,
                                     N_EPOCHS, PATIENCE, FINETUNE_LR_SCALE)

OUT_JSON = Path('results/grid_summary.json')

ARMS = [
    ('weight-only',   lambda b: (b, None)),
    ('membrane-only', lambda b: (None, b)),
    ('joint',         lambda b: (b, b)),
]
BIT_WIDTHS = [8, 4, 2]


def discover_seeds(input_size):
    # Seeds for which an FP32 checkpoint exists in ascending order
    seeds = []
    for path in RUNS_DIR.glob(f'fp32_{input_size}px_seed*'):
        match = re.fullmatch(rf'fp32_{input_size}px_seed(\d+)', path.name)
        if match and (path / 'best_model.pt').exists():
            seeds.append(int(match.group(1)))
    return sorted(seeds)


def build_plan(seeds, input_size):
    # runs to perform, seed-major, control first then 8 -> 4 -> 2

    plan = []
    for seed in seeds:
        plan.append({
            'arm': 'fp32-control',
            'bits': None,
            'weight_bits': None,
            'membrane_bits': None,
            'control': True,
            'seed': seed,
            'tag': config_tag(None, None, input_size, seed),
        })
        for bits in BIT_WIDTHS:
            for arm_name, make in ARMS:
                weight_bits, membrane_bits = make(bits)
                plan.append({
                    'arm': arm_name,
                    'bits': bits,
                    'weight_bits': weight_bits,
                    'membrane_bits': membrane_bits,
                    'control': False,
                    'seed': seed,
                    'tag': config_tag(weight_bits, membrane_bits, input_size, seed),
                })
    return plan


def already_done(tag):
    return (RUNS_DIR / tag / 'run.json').exists()


def format_duration(seconds):
    return str(timedelta(seconds=int(seconds)))

def main():
    parser = argparse.ArgumentParser(
        description='Run the quantisation grid: 3 arms x 3 bit-widths x seeds')
    parser.add_argument('--input-size', type=int, default=INPUT_SIZE)
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='default: every seed with an FP32 checkpoint')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--lr-scale', type=float, default=FINETUNE_LR_SCALE)
    parser.add_argument('--dry-run', action='store_true',
                        help='list the runs and stop without training anything')
    args = parser.parse_args()

    input_size = args.input_size
    seeds = args.seeds if args.seeds is not None else discover_seeds(input_size)

    if not seeds:
        raise SystemExit(
            f'No FP32 checkpoint found under {RUNS_DIR}/fp32_{input_size}px_seed*.\n')

    plan = build_plan(seeds, input_size)
    pending = [run for run in plan if not already_done(run['tag'])]
    done = len(plan) - len(pending)

    print(f'grid: ({len(ARMS)} arms x {len(BIT_WIDTHS)} bit-widths + 1 FP32 control) '
          f'x {len(seeds)} seeds = {len(plan)} runs')
    print(f'seeds {seeds} | {input_size} px | epochs {args.epochs} | '
          f'patience {args.patience} | lr scale {args.lr_scale:g}')
    if done:
        print(f'{done} already present, {len(pending)} to run')

    missing = sorted({str(fp32_checkpoint_for(input_size, run['seed']))
                      for run in pending
                      if not fp32_checkpoint_for(input_size, run['seed']).exists()})
    if missing:
        print('\nmissing FP32 checkpoints:')
        for path in missing:
            print(f'  {path}')
        raise SystemExit('aborting before the grid starts')

    if args.dry_run:
        print('\nwould run in this order:')
        for i, run in enumerate(pending, start=1):
            bits = f'{run["bits"]}b' if run['bits'] else '-'
            print(f'  {i:>3}. {run["tag"]:<30} {run["arm"]:<14} {bits}')
        return

    hyperparams = load_hyperparams()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device {device}\n')

    results = {}
    failures = []
    durations = []
    started = time.time()

    for i, run in enumerate(pending, start=1):
        elapsed = time.time() - started
        if durations:
            remaining = (sum(durations) / len(durations)) * (len(pending) - i + 1)
            eta = f' | elapsed {format_duration(elapsed)}, ~{format_duration(remaining)} left'
        else:
            eta = ''
        print(f'[{i}/{len(pending)}] {run["tag"]}{eta}', flush=True)

        run_started = time.time()
        try:
            summary = run_qat(input_size, run['seed'], hyperparams,
                              weight_bits=run['weight_bits'],
                              membrane_bits=run['membrane_bits'],
                              n_epochs=args.epochs, patience=args.patience,
                              lr_scale=args.lr_scale, device=device, verbose=True,
                              control=run['control'])
            results[run['tag']] = {
                'arm': run['arm'],
                'bits': run['bits'],
                'seed': run['seed'],
                'ptq_accuracy': summary['ptq_accuracy'],
                'qat_accuracy': summary['best_val_accuracy'],
                'fp32_start_accuracy': summary['fp32_start_accuracy'],
                'recovered_over_ptq': summary['recovered_over_ptq'],
                'epochs_run': summary['epochs_run'],
                'stopped_early': summary['stopped_early'],
                'checkpoint_firing_rates': summary['checkpoint_firing_rates'],
            }
            durations.append(time.time() - run_started)
        except Exception as error:          # noqa: BLE001: one bad run must not end the grid
            print(f'  FAILED: {error}', flush=True)
            traceback.print_exc()
            failures.append({'tag': run['tag'], 'error': repr(error)})

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'input_size': input_size,
                   'seeds': seeds,
                   'epochs': args.epochs,
                   'patience': args.patience,
                   'lr_scale': args.lr_scale,
                   'total_seconds': time.time() - started,
                   'failures': failures,
                   'results': results}, f, indent=2)

    # Report
    print(f'\n{"=" * 78}')
    print(f'{len(results)} runs completed in {format_duration(time.time() - started)}'
          + (f', {len(failures)} failed' if failures else ''))

    if results:
        # Grouped by seed
        print(f'\n{"config":<16}{"seed":>7}{"PTQ":>9}{"QAT":>9}{"recovered":>12}'
              f'{"epochs":>9}{"rate L3":>10}')
        print('-' * 78)
        for tag, r in results.items():
            n_val = 1435
            recovered = r['recovered_over_ptq'] * n_val
            label = f'{r["arm"]} {r["bits"]}b' if r['bits'] else r['arm']
            print(f'{label:<16}{r["seed"]:>7}{r["ptq_accuracy"]:>9.4f}'
                  f'{r["qat_accuracy"]:>9.4f}{recovered:>+12.0f}'
                  f'{r["epochs_run"]:>9}'
                  f'{r["checkpoint_firing_rates"]["layer3"]:>10.4f}')

    if failures:
        print('\nfailed:')
        for failure in failures:
            print(f'  {failure["tag"]}: {failure["error"]}')

    print(f'\nwritten to {OUT_JSON}')

if __name__ == '__main__':
    main()