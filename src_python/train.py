'''
Final training with the best hyperparameters found by Optuna.
Parametrized on resoltion and seed
'''

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.append(os.path.abspath('C:/Users/devam/OneDrive/Tesi'))
from models.snn_model import GWGlitchSNN
from src_python.dataloader import build_dataloaders, INPUT_SIZE
from src_python.training_utils import train_one_epoch, validate, validate_per_class

HYPERPARAMS_FILE = 'best_hyperparams.json'
RUNS_DIR = Path('runs')
N_EPOCHS = 50
PATIENCE = 5


def load_hyperparams(path=HYPERPARAMS_FILE):
    with open(path) as f:
        return json.load(f)


def run_training(input_size, seed, hyperparams, tag=None, n_epochs=N_EPOCHS,
                 patience=PATIENCE, device=None, verbose=True):
    
    # Train a configuration and returns the summary of the run
    tag = tag or f'fp32_{input_size}px_seed{seed}'
    out_dir = RUNS_DIR / tag
    summary_path = out_dir / 'run.json'

    if summary_path.exists():
        if verbose:
            print(f'[{tag}] already present: skipping')
        with open(summary_path) as f:
            return json.load(f)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / 'best_model.pt'

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset, train_dl, val_dl, _ = build_dataloaders(input_size=input_size, verbose=False)

    learning_rate = hyperparams["learning_rate"]
    beta = hyperparams["beta"]
    time_steps = hyperparams["time_steps"]

    net = GWGlitchSNN(beta=beta, input_size=input_size).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    if verbose:
        print(f'\n=== {tag} ===')
        print(f'  resolution {input_size} px | seed {seed} | T={time_steps} '
              f'| beta={beta:.4f} | lr={learning_rate:.2e}')
        print(f'  train {len(train_dl.dataset)} | val {len(val_dl.dataset)} '
              f'| parameters {sum(p.numel() for p in net.parameters()):,}')

    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    stopped_early = False
    history = []
    epoch = -1

    for epoch in range(n_epochs):
        avg_firing_rate, layer_firing_rates = train_one_epoch(
            net, train_dl, optimizer, criterion, time_steps, device)
        val_loss, val_accuracy = validate(net, val_dl, criterion, time_steps, device)

        history.append({'epoch': epoch, 'val_loss': val_loss,
                        'val_accuracy': val_accuracy,
                        'avg_firing_rate': avg_firing_rate,
                        'layer_firing_rates': layer_firing_rates})

        if verbose:
            print(f'  Epoch {epoch}: val_loss={val_loss:.4f}, '
                  f'val_accuracy={val_accuracy:.4f}, '
                  f'avg_firing_rate={avg_firing_rate:.4f}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(net.state_dict(), checkpoint)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f'early stopping: {patience} epochs w/o improvement of validation loss')
            stopped_early = True
            break

    # recall per class on best checkpoint
    net.load_state_dict(torch.load(checkpoint, map_location=device))
    recall, support = validate_per_class(net, val_dl, time_steps, device,
                                         n_classes=len(dataset.classes))
    per_class = {name: r for name, r in zip(dataset.classes, recall)}

    if verbose:
        print(f'  best val accuracy {best_val_accuracy:.4f} (epoch {best_epoch})')
        for name, r in per_class.items():
            print(f'    {name:<18} recall {r:.4f}')

    summary = {
        'tag': tag,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'input_size': input_size,
        'seed': seed,
        'hyperparameters': hyperparams,
        'n_train': len(train_dl.dataset),
        'n_val': len(val_dl.dataset),
        'n_parameters': sum(p.numel() for p in net.parameters()),
        'best_epoch': best_epoch,
        'best_val_accuracy': best_val_accuracy,
        'best_val_loss': best_val_loss,
        'val_recall_per_class': per_class,
        'val_support_per_class': dict(zip(dataset.classes, support)),
        'final_firing_rates': history[-1]['layer_firing_rates'],
        'epochs_run': epoch + 1,
        'stopped_early': stopped_early,
        'checkpoint': str(checkpoint),
        'history': history,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    del net, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', type=int, default=INPUT_SIZE)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=N_EPOCHS)
    args = parser.parse_args()

    hyperparams = load_hyperparams()
    print('Best hyperparameters:', hyperparams)

    summary = run_training(args.input_size, args.seed, hyperparams,
                           tag=args.tag, n_epochs=args.epochs)
    print(f"\nSummary in/{summary['tag']}/run.json")


if __name__ == "__main__":
    main()