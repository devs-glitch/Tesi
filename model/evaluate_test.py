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
from scripts.dataloader import test_dataloader, INPUT_SIZE
from scripts.training_utils import direct_encode, validate
from scripts.per_sample import loader_paths, save_per_sample

PER_SAMPLE_CSV = 'results/test_predictions.csv'


def predict_per_sample(net, dataloader, time_steps, device):
    # firing rate prediction per sample
    net.eval()
    preds, trues, rates = [], [], []
    for images, labels in dataloader:
        with torch.no_grad():
            images = images.to(device)
            spike_out, s1, s2, s3, s4 = net(direct_encode(images, time_steps))
            preds.extend(torch.argmax(spike_out.sum(dim=0), dim=1).cpu().tolist())
            trues.extend(labels.tolist())
            # mean on time-stept and dimensions (not batch, dim=1)
            per_layer = [s.mean(dim=(0, 2, 3, 4)).cpu() for s in (s1, s2, s3, s4)]
            rates.extend(zip(*[t.tolist() for t in per_layer]))
    return preds, trues, rates


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("best_hyperparams.json", "r") as f:
        best_params = json.load(f)

    net = GWGlitchSNN(beta=best_params["beta"], input_size=INPUT_SIZE).to(device)
    net.load_state_dict(torch.load("best_model.pt", map_location=device))
    net.eval()

    time_steps = best_params["time_steps"]
    criterion = torch.nn.CrossEntropyLoss()

    test_loss, test_accuracy = validate(net, test_dataloader, criterion, time_steps, device)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # output per sample
    classes = test_dataloader.dataset.dataset.classes
    paths = loader_paths(test_dataloader)
    preds, trues, rates = predict_per_sample(net, test_dataloader, time_steps, device)

    assert len(paths) == len(preds), 'disallagniment btw path and predictions'

    rows = [{'path': p,
             'true': classes[t],
             'predicted': classes[q],
             'correct': int(t == q),
             'firing_layer1': f'{r[0]:.6f}',
             'firing_layer2': f'{r[1]:.6f}',
             'firing_layer3': f'{r[2]:.6f}',
             'firing_layer4': f'{r[3]:.6f}',
             'firing_mean': f'{sum(r) / len(r):.6f}'}
            for p, t, q, r in zip(paths, trues, preds, rates)]
    save_per_sample(rows, PER_SAMPLE_CSV)

    agreement = sum(r['correct'] for r in rows) / len(rows)
    assert abs(agreement - test_accuracy) < 1e-6, \
        'per sample predictio do not reproduced the aggregated accuracy'


if __name__ == "__main__":
    main()
