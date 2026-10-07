'''
Final evaluation on the test set. Every model, one single opening
'''
import argparse
import json
import time
from pathlib import Path

import torch

from scripts.dataloader import build_dataloaders
from scripts.evaluate_model import default_models, load_model
from scripts.per_sample import loader_paths, save_per_sample
from scripts.training_utils import direct_encode, validate
from xai.sam import MANIFEST

OUT_DIR = Path('results/test')
SUMMARY_JSON = OUT_DIR / 'test_evaluation.json'
AUDIT_JSON = Path('results/test_opened.json')


def wilson_interval(successes, total, z=1.959963984540054):
    # wilson score interval for a binomial proportion - 95% by default
    if total == 0:
        return float('nan'), float('nan')
    p = successes / total
    denom = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denom
    half = z * ((p * (1 - p) / total + z ** 2 / (4 * total ** 2)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def predict_per_sample(net, dataloader, time_steps, device):
    # per-image prediction plus per-layer firing rate
    net.eval()
    preds, trues, rates = [], [], []
    for images, labels in dataloader:
        with torch.no_grad():
            images = images.to(device)
            spike_out, s1, s2, s3, s4 = net(direct_encode(images, time_steps))
            preds.extend(torch.argmax(spike_out.sum(dim=0), dim=1).cpu().tolist())
            trues.extend(labels.tolist())
            # mean on time-step and dimensions (not batch, dim=1)
            per_layer = [s.mean(dim=(0, 2, 3, 4)).cpu() for s in (s1, s2, s3, s4)]
            rates.extend(zip(*[t.tolist() for t in per_layer]))
    return preds, trues, rates

def evaluate_one(net, time_steps, device, dataloader, paths, classes, criterion):
    # the same two-route check as before for one model
    test_loss, test_accuracy = validate(net, dataloader, criterion,
                                        time_steps, device)
    preds, trues, rates = predict_per_sample(net, dataloader, time_steps, device)
    assert len(paths) == len(preds), 'path and prediction lists are misaligned'

    rows = [{'path': p,
             'true': classes[t],
             'predicted': classes[q],
             'correct': int(t == q),
             'firing_layer1': f'{r[0]:.6f}',
             'firing_layer2': f'{r[1]:.6f}',
             'firing_layer3': f'{r[2]:.6f}',
             'firing_layer4': f'{r[3]:.6f}'}
            for p, t, q, r in zip(paths, trues, preds, rates)]

    n = len(rows)
    correct = sum(r['correct'] for r in rows)
    assert abs(correct / n - test_accuracy) < 1e-6, \
        'per-sample predictions do not reproduce the aggregate accuracy'

    lo, hi = wilson_interval(correct, n)
    summary = {'loss': test_loss, 'accuracy': test_accuracy, 'n': n,
               'correct': correct, 'wilson_95': [lo, hi], 'per_class': {}}
    for c, name in enumerate(classes):
        subset = [r for r, t in zip(rows, trues) if t == c]
        k = sum(r['correct'] for r in subset)
        clo, chi = wilson_interval(k, len(subset))
        summary['per_class'][name] = {'recall': k / len(subset),
                                      'support': len(subset),
                                      'wilson_95': [clo, chi]}
    return summary, rows


def preflight(names):
    # verify every checkpoint before the test loader is touched
    missing = []
    for name in names:
        run = Path('runs') / name
        missing += [str(run / f) for f in ('run.json', 'best_model.pt')
                    if not (run / f).exists()]
    if missing:
        raise SystemExit('cannot start, these files are missing:\n  '
                         + '\n  '.join(missing))


def previous_openings():
    if not AUDIT_JSON.exists():
        return []
    with open(AUDIT_JSON, encoding='utf-8') as f:
        return json.load(f).get('openings', [])


def record_opening(models):
    openings = previous_openings()
    openings.append({'when': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                     'n_models': len(models), 'models': models})
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'openings': openings}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Open the test partition once, on every model of the grid')
    parser.add_argument('models', nargs='*', default=None,
                        help='run directory names; default: every run at the manifest resolution')
    parser.add_argument('--force', action='store_true',
                        help='open the partition again. Recorded as a second '
                             'opening in results/test_opened.json')
    args = parser.parse_args()

    if SUMMARY_JSON.exists() and not args.force:
        earlier = previous_openings()
        detail = (f'\nPrevious openings recorded in {AUDIT_JSON}:\n' +
                  '\n'.join(f'  {o["when"]}  {o["n_models"]} models'
                            for o in earlier)) if earlier else ''
        raise SystemExit(
            f'{SUMMARY_JSON} already exists: the test partition has been opened'
            f'{detail}\n\nEvery further look is taken knowing the first one')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(MANIFEST, encoding='utf-8') as f:
        manifest = json.load(f)
    input_size = manifest['input_size']

    names = args.models or default_models(input_size)
    if not names:
        raise SystemExit('no runs found at the manifest resolution')

    preflight(names)

    dataset, _, _, test_dataloader = build_dataloaders(input_size=input_size,
                                                       verbose=False)
    classes = dataset.classes
    criterion = torch.nn.CrossEntropyLoss()
    paths = loader_paths(test_dataloader)

    print('=' * 72)
    print(f'Opening test partition|  {len(paths)} images  |  {input_size} px')
    print(f'{len(names)} models in one pass')
    print('=' * 72)

    results = {}
    header = (f'\n{"model":<28}{"accuracy":>10}{"correct":>9}'
              f'{"95% Wilson":>22}{"loss":>9}')
    print(header)
    print('-' * (len(header) - 1))

    for name in names:
        net, handle, time_steps, model_size, meta = load_model(name, device)
        if model_size != input_size:
            if handle is not None:
                handle.remove()
            raise SystemExit(
                f'{name} was built for {model_size} px, manifest says '
                f'{input_size} px: the test loader would feed it the wrong size')

        summary, rows = evaluate_one(net, time_steps, device, test_dataloader,
                                     paths, classes, criterion)
        if handle is not None:
            handle.remove()

        summary.update({
            'arm': meta.get('arm'),
            'weight_bits': (meta.get('quantisation') or {}).get('weight_bits'),
            'membrane_bits': (meta.get('quantisation') or {}).get('membrane_bits'),
            'seed': meta.get('seed'),
        })
        results[name] = summary
        save_per_sample(rows, OUT_DIR / f'{name}_test_predictions.csv')

        lo, hi = summary['wilson_95']
        print(f'{name:<28}{summary["accuracy"]:>10.4f}'
              f'{summary["correct"]:>5}/{summary["n"]:<4}'
              f'{f"[{lo:.4f}, {hi:.4f}]":>22}{summary["loss"]:>9.4f}')

    # per class
    print(f'\n{"model":<28}' + ''.join(f'{c[:14]:>16}' for c in classes)
          + '    (recall)')
    for name, s in results.items():
        print(f'{name:<28}'
              + ''.join(f'{s["per_class"][c]["recall"]:>16.4f}' for c in classes))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_JSON, 'w', encoding='utf-8') as f:
        json.dump({'input_size': input_size, 'n_test': len(paths),
                   'classes': list(classes), 'models': results}, f, indent=2)
    record_opening(names)

    print(f'\nsummary:     {SUMMARY_JSON}')
    print(f'per-sample:  {OUT_DIR}/<model>_test_predictions.csv')
    print(f'audit trail: {AUDIT_JSON}')


if __name__ == '__main__':
    main()