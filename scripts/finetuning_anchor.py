'''
The fine-tuning anchor: how far continued training alone moves the model
'''
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from model.snn_model import GWGlitchSNN
from scripts.dataloader import build_dataloaders
from scripts.evaluate_model import evaluate
from xai.divergence_metrics import compare_models, load_fp32_for_seed
from xai.sam import MANIFEST, load_reference_model
from xai.sam_metrics import samples_per_class

RUNS_DIR = Path('runs')
OUT_JSON = Path('results/finetuning_anchor.json')
OUT_CSV = Path('results/tables/finetuning_anchor.csv')
N_PER_CLASS = 20    
TOP_K = 0.20


def control_runs(input_size):
    # control arm only: weights and membrane both unquantised
    found = []
    for run in sorted(RUNS_DIR.glob(f'qat_wxux_{input_size}px_seed*')):
        if (run / 'run.json').exists():
            found.append(run)
    return found


def weight_distance(a, b):
    # Per-tensor distance between two state dicts, in weight space
    if set(a) != set(b):
        raise ValueError('the two checkpoints do not hold the same tensors')
    out = {}
    for key in sorted(a):
        x, y = a[key].double(), b[key].double()
        delta = (y - x).flatten()
        norm = float(x.norm())
        out[key] = {
            'n_params': int(delta.numel()),
            'l2': float(delta.norm()),
            'max_abs': float(delta.abs().max()),
            'relative_l2': float(delta.norm() / norm) if norm > 0 else float('nan'),
        }
    total = float(torch.cat([(b[k].double() - a[k].double()).flatten()
                             for k in sorted(a)]).norm())
    return out, total


def load_plain(path, beta, input_size, device):
    # The control arm quantises nothing - no quantiser is attached
    net = GWGlitchSNN(beta=beta, input_size=input_size).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net


def summarise(divergence, classes):
    iou = [divergence[c]['iou'] for c in classes]
    pcc = [divergence[c]['pcc'] for c in classes]
    return float(np.mean(iou)), float(np.mean(pcc)), float(min(iou))


def main():
    parser = argparse.ArgumentParser(
        description='Divergence and health of the control arm after fine-tuning '
                    'against the full-precision parent (validation only)')
    parser.add_argument('--n-per-class', type=int, default=N_PER_CLASS)
    parser.add_argument('--force', action='store_true',
                        help='overwrite an existing result file')
    args = parser.parse_args()

    if OUT_JSON.exists() and not args.force:
        raise SystemExit(f'{OUT_JSON} already exists. Pass --force to replace it')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reference_net, time_steps, input_size, layer_index, gamma = load_reference_model(
        device, MANIFEST)
    del reference_net   # the per-seed parent is loaded inside the loop

    runs = control_runs(input_size)
    if not runs:
        raise SystemExit(f'no control runs under {RUNS_DIR}/qat_wxux_{input_size}px_seed*')

    missing = [str(r / f) for r in runs for f in ('best_model.pt', 'final_model.pt')
               if not (r / f).exists()]
    if missing:
        raise SystemExit('cannot start, these checkpoints are missing:\n  '
                         + '\n  '.join(missing))

    # The fourth loader is the test partition. It is bound to _ and never used
    dataset, _, val_dataloader, _ = build_dataloaders(input_size=input_size,
                                                      verbose=False)
    classes = dataset.classes
    samples = samples_per_class(val_dataloader, len(classes), args.n_per_class)

    with open(MANIFEST, encoding='utf-8') as f:
        beta = json.load(f)['hyperparameters']['beta']

    print('=' * 74)
    print(f'FINE-TUNING ANCHOR  |  {input_size} px  |  T={time_steps}  |  '
          f'layer {layer_index}  |  gamma {gamma}')
    print(f'{len(runs)} control runs, {args.n_per_class} validation images per class')
    print('=' * 74)

    output = {'input_size': input_size, 'layer': layer_index, 'gamma': gamma,
              'n_per_class': args.n_per_class, 'top_k': TOP_K,
              'partition': 'validation', 'runs': {}}
    rows = []

    for run in runs:
        with open(run / 'run.json', encoding='utf-8') as f:
            summary = json.load(f)
        quant = summary.get('quantisation') or {}
        if quant.get('weight_bits') or quant.get('membrane_bits'):
            raise SystemExit(f'{run.name} is not a control run: {quant}')
        seed = summary['seed']

        # ---- weight space: did fine-tuning move the parameters at all
        selected = torch.load(run / 'best_model.pt', map_location='cpu')
        final = torch.load(run / 'final_model.pt', map_location='cpu')
        per_tensor, total_l2 = weight_distance(selected, final)
        moved = total_l2 > 0

        print(f'\n--- {run.name}   seed {seed}')
        print(f'  epochs run {summary.get("epochs_run", "?")}, '
              f'best epoch {summary.get("best_epoch", "?")}, '
              f'recovered over the starting point '
              f'{summary.get("recovered_over_ptq", float("nan")):+.6f}')
        # best_epoch == -1 is the direct record of what happened, -1 means no epoch ever did and the saved checkpoint is still the parent
        if summary.get('best_epoch') == -1:
            print('  best epoch -1: no epoch beat the starting candidate, so the '
                  'selected checkpoint is the parent')
        print(f'  |final - selected| over all parameters: {total_l2:.6g}')
        if not moved:
            print('  the two checkpoints are the same weights: fine-tuning left '
                  'the parameters untouched, and there is no anchor to measure')

        parent = load_fp32_for_seed(seed, input_size, device)
        entry = {'seed': seed,
                 'epochs_run': summary.get('epochs_run'),
                 'best_epoch': summary.get('best_epoch'),
                 'recovered_over_ptq': summary.get('recovered_over_ptq'),
                 'best_val_accuracy': summary.get('best_val_accuracy'),
                 'weight_delta_l2': total_l2,
                 'weight_delta_per_tensor': per_tensor,
                 'checkpoints': {}}

        for which, filename in (('selected', 'best_model.pt'),
                                ('final', 'final_model.pt')):
            net = load_plain(run / filename, beta, input_size, device)
            divergence = compare_models(parent, net, samples, classes,
                                        time_steps, device, layer_index, gamma,
                                        use_ssim=False)
            mean_iou, mean_pcc, worst_iou = summarise(divergence, classes)
            record = {'file': filename, 'divergence': divergence,
                      'mean_iou': mean_iou, 'mean_pcc': mean_pcc,
                      'worst_iou': worst_iou}

            if which == 'final':
                aggregates, _, _ = evaluate(net, val_dataloader, time_steps, device)
                record.update({
                    'accuracy': aggregates['accuracy'],
                    'mean_rate': aggregates['mean_rate'],
                    'dead_fraction': aggregates['dead_fraction'],
                    'saturated_fraction': aggregates['saturated_fraction'],
                    'cv_isi': aggregates['cv_isi'],
                    'cv_isi_defined_fraction': aggregates['cv_isi_defined_fraction'],
                })
            else:
                aggregates = {'accuracy': summary['best_val_accuracy'],
                              'mean_rate': {}}
                record['accuracy'] = aggregates['accuracy']
                record['accuracy_source'] = 'run.json best_val_accuracy'
            entry['checkpoints'][which] = record
            print(f'  {which:<9} acc {aggregates["accuracy"]:.4f}   '
                  f'IoU {mean_iou:.4f}   PCC {mean_pcc:.4f}   '
                  f'worst class IoU {worst_iou:.4f}')

            # run.json already carries the firing rates measured on the final model at the end of training
            if which == 'final':
                recorded = summary.get('final_firing_rates') or {}
                for index, value in aggregates['mean_rate'].items():
                    expected = recorded.get(f'layer{index}')
                    if expected is not None and abs(expected - value) > 1e-6:
                        print(f'    WARNING layer {index}: measured {value:.6f}, '
                              f'run.json recorded {expected:.6f}')
            for cls in classes:
                rows.append({'run': run.name, 'seed': seed, 'checkpoint': which,
                             'class': cls,
                             'n_images': divergence[cls]['n'],
                             'iou': divergence[cls]['iou'],
                             'pcc': divergence[cls]['pcc'],
                             'pcc_std': divergence[cls]['pcc_std'],
                             'com_shift': divergence[cls]['com_shift'],
                             'com_shift_abs': divergence[cls]['com_shift_abs'],
                             'accuracy': aggregates['accuracy']})

        # the selected checkpoint is the parent, so its divergence must be exactly 1
        selected_iou = entry['checkpoints']['selected']['mean_iou']
        if abs(selected_iou - 1.0) > 1e-9:
            raise SystemExit(
                f'{run.name}: the selected checkpoint diverges from the parent '
                f'(IoU {selected_iou:.6f}, must be 1.000000)')

        output['runs'][run.name] = entry

    finals = [v['checkpoints']['final'] for v in output['runs'].values()]
    anchor = {
        'mean_iou': float(np.mean([f['mean_iou'] for f in finals])),
        'min_iou': float(min(f['mean_iou'] for f in finals)),
        'max_iou': float(max(f['mean_iou'] for f in finals)),
        'mean_pcc': float(np.mean([f['mean_pcc'] for f in finals])),
        'worst_class_iou': float(min(f['worst_iou'] for f in finals)),
        'mean_accuracy': float(np.mean([f['accuracy'] for f in finals])),
    }
    output['anchor'] = anchor

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print('\n' + '=' * 74)
    print('Anchor: fine-tuning alone, no bits removed')
    print(f'  top-20 % overlap with the parent  {anchor["mean_iou"]:.4f}  '
          f'[{anchor["min_iou"]:.4f}, {anchor["max_iou"]:.4f}] over three seeds')
    print(f'  Pearson correlation               {anchor["mean_pcc"]:.4f}')
    print(f'  worst class and seed              {anchor["worst_class_iou"]:.4f}')
    print(f'  validation accuracy               {anchor["mean_accuracy"]:.4f}')
    print('\nRead against the other two anchors in divergence.json: the numerical')
    print(f'\nwritten to {OUT_JSON}\n            {OUT_CSV}')


if __name__ == '__main__':
    main()