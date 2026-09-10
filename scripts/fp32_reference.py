'''
FP32 reference with final configuration
confont divergences from seed to seed 
writes baseline_manifest.json
'''

import hashlib
import json
import statistics as st
from datetime import datetime
from pathlib import Path

import torch

from scripts.train import load_hyperparams, run_training

SEEDS = (1234, 2345, 3456)
MANIFEST = Path('baseline_manifest.json')
SPLIT_FILE = Path('data/split_assignment.csv')

SAM_REFERENCE_LAYER = 2
SAM_GAMMA = None


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def agg(values):
    return {'mean': st.mean(values),
            'std': st.stdev(values) if len(values) > 1 else 0.0,
            'values': values}


def main():
    hp = load_hyperparams()
    input_size = hp['input_size']
    print(f'FP32 reference: {input_size} px | beta={hp["beta"]} | '
          f'T={hp["time_steps"]} | lr={hp["learning_rate"]:.3e}')
    print(f'Seed: {SEEDS}\n')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = [run_training(input_size, s, hp, device=device) for s in SEEDS]

    classes = sorted(runs[0]['val_recall_per_class'])
    layers = ('layer1', 'layer2', 'layer3', 'layer4')

    summary = {
        'val_accuracy': agg([r['best_val_accuracy'] for r in runs]),
        'epochs_run': agg([float(r['epochs_run']) for r in runs]),
        'val_recall_per_class': {c: agg([r['val_recall_per_class'][c] for r in runs])
                                 for c in classes},
        'checkpoint_firing_rates': {l: agg([r['checkpoint_firing_rates'][l] for r in runs])
                                    for l in layers},
    }

    print(f'{"":<26}{"mean":>12}{"dev.std.":>12}{"range":>12}')

    def line(label, a, fmt='{:.4f}'):
        rng = f'{max(a["values"]) - min(a["values"]):.4f}'
        print(f'{label:<26}{fmt.format(a["mean"]):>12}{fmt.format(a["std"]):>12}{rng:>12}')

    line('validation accuracy', summary['val_accuracy'])
    line('epochs', summary['epochs_run'], '{:.1f}')
    for c in classes:
        line(f'  recall {c}', summary['val_recall_per_class'][c])
    for l in layers:
        line(f'  firing rate {l}', summary['checkpoint_firing_rates'][l])

    manifest = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'input_size': input_size,
        'hyperparameters': {k: hp[k] for k in ('learning_rate', 'beta', 'time_steps')},
        'hyperparameter_study': hp.get('study_name'),
        'search_space': hp.get('search_space'),
        'sam': {'reference_layer': SAM_REFERENCE_LAYER,
                'gamma': SAM_GAMMA},
        'seeds': list(SEEDS),
        'split_file': str(SPLIT_FILE),
        'split_sha256_16': file_digest(SPLIT_FILE),
        'n_train': runs[0]['n_train'],
        'n_val': runs[0]['n_val'],
        'n_parameters': runs[0]['n_parameters'],
        'checkpoints': [r['checkpoint'] for r in runs],
        'reference': summary,
    }
    with open(MANIFEST, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\nConfiguration in {MANIFEST}')
    print(f'{SPLIT_FILE}: {manifest["split_sha256_16"]}')


if __name__ == '__main__':
    main()
