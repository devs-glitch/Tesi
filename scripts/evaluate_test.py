'''
Final evaluation of model on test set

We compute accuracy in two different ways:
1) validate() on the test set, returning an aggregated value
2) predict_per_sample() makes prediction for each sample: the fraction of correct predictions needs to be the same number
'''
import os
import sys
import json

import torch

from model.snn_model import GWGlitchSNN
from scripts.dataloader import build_dataloaders
from scripts.training_utils import direct_encode, validate
from scripts.per_sample import loader_paths, save_per_sample
from xai.sam import load_reference_model

PER_SAMPLE_CSV = 'results/test_predictions.csv'
SUMMARY_JSON = 'results/test_evaluation.json'

def wilson_interval(successes, total, z=1.959963984540054):
    # wilson score interval for a binomial proportion, 95% by default
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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, time_steps, input_size, layer_index, gamma = load_reference_model(
        device)
 
    print(f'{input_size} px | T={time_steps} | layer {layer_index} | gamma {gamma}')
    print('\nOpening test partition\n')
 
    dataset, _, _, test_dataloader = build_dataloaders(input_size=input_size,
                                                       verbose=False)
    classes = dataset.classes
    criterion = torch.nn.CrossEntropyLoss()
 
    test_loss, test_accuracy = validate(net, test_dataloader, criterion,
                                        time_steps, device)
 
    paths = loader_paths(test_dataloader)
    preds, trues, rates = predict_per_sample(net, test_dataloader, time_steps,
                                             device)
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
    save_per_sample(rows, PER_SAMPLE_CSV)
 
    n = len(rows)
    correct = sum(r['correct'] for r in rows)
    assert abs(correct / n - test_accuracy) < 1e-6, \
        'per-sample predictions do not reproduce the aggregate accuracy'
 
    lo, hi = wilson_interval(correct, n)
    print(f'Test loss:     {test_loss:.4f}')
    print(f'Test accuracy: {test_accuracy:.4f}  '
          f'({correct}/{n}, 95% Wilson [{lo:.4f}, {hi:.4f}])')
 
    summary = {'loss': test_loss, 'accuracy': test_accuracy, 'n': n,
               'correct': correct, 'wilson_95': [lo, hi], 'per_class': {}}
 
    print(f'\n{"class":<18}{"recall":>9}{"support":>9}{"95% Wilson":>22}')
    for c, name in enumerate(classes):
        subset = [r for r, t in zip(rows, trues) if t == c]
        k = sum(r['correct'] for r in subset)
        clo, chi = wilson_interval(k, len(subset))
        summary['per_class'][name] = {'recall': k / len(subset),
                                      'support': len(subset),
                                      'wilson_95': [clo, chi]}
        print(f'{name:<18}{k / len(subset):>9.4f}{len(subset):>9}'
              f'{f"[{clo:.4f}, {chi:.4f}]":>22}')
 
    with open(SUMMARY_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'\nper-sample CSV: {PER_SAMPLE_CSV}')
    print(f'summary:        {SUMMARY_JSON}')
 
 
if __name__ == '__main__':
    main()
 