'''
Study on resolution of images: 224, 112, 64 px, for tree seeds each
Hyperparameters are the one fixed at 224 px. If 64 px (hypothized goal) keeps up
after re-optimization it will only go better.
Tests run on the validation set
'''

import csv
import json
import statistics
from pathlib import Path

import torch

from src_python.train1 import load_hyperparams, run_training

RESOLUTIONS = (224, 112, 64)
SEEDS = (1234, 2345, 3456)
RESULTS_CSV = Path('runs/resolution_study.csv')


def main():
    hyperparams = load_hyperparams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Hyperparameters: {hyperparams}')
    print(f'Device: {device}')

    results = []
    for input_size in RESOLUTIONS:
        for seed in SEEDS:
            summary = run_training(input_size, seed, hyperparams, device=device)
            results.append(summary)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    classes = sorted(results[0]['val_recall_per_class'])
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['input_size', 'seed', 'val_accuracy', 'best_epoch',
                         'epochs_run', 'n_parameters']
                        + [f'recall_{c}' for c in classes]
                        + [f'firing_{l}' for l in ('layer1', 'layer2', 'layer3', 'layer4')])
        for r in results:
            writer.writerow(
                [r['input_size'], r['seed'], f"{r['best_val_accuracy']:.4f}",
                 r['best_epoch'], r['epochs_run'], r['n_parameters']]
                + [f"{r['val_recall_per_class'][c]:.4f}" for c in classes]
                + [f"{r['final_firing_rates'][l]:.4f}"
                   for l in ('layer1', 'layer2', 'layer3', 'layer4')])

    print(f'\nRaw results in {RESULTS_CSV}\n')
    print('=' * 78)
    print('Validation accuracy (mean +- dev.std. 3 seeds)')
    print('=' * 78)
    header = f'{"Resolution":<14}{"Accuracy":<20}' + ''.join(f'{c[:12]:>15}' for c in classes)
    print(header)
    print('-' * len(header))

    for input_size in RESOLUTIONS:
        runs = [r for r in results if r['input_size'] == input_size]
        accs = [r['best_val_accuracy'] for r in runs]
        row = f'{str(input_size) + " px":<14}'
        row += f'{statistics.mean(accs):.4f} +- {statistics.stdev(accs):.4f}   '
        for c in classes:
            vals = [r['val_recall_per_class'][c] for r in runs]
            row += f'{statistics.mean(vals):>10.4f}     '
        print(row)


if __name__ == '__main__':
    main()