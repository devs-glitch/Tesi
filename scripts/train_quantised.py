'''
Quantisation-aware fine-tuning
Fine-tuning starts from the FP32 checkpoint of the same seed
'''

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.nn.utils import parametrize

from model.quantization import apply_quantisation, MEMBRANE_RANGE
from model.snn_model import GWGlitchSNN
from scripts.dataloader import build_dataloaders
from scripts.train import load_hyperparams, RUNS_DIR, INPUT_SIZE
from scripts.training_utils import (train_one_epoch, validate,
                                    validate_per_class, measure_firing_rates)

# Fine-tuning not training: the model starts from a solution

FINETUNE_LR_SCALE = 0.1
N_EPOCHS = 20
PATIENCE = 5

ACCURACY_TOLERANCE = 4 / 1435


def arm_of(weight_bits, membrane_bits):
    if weight_bits and membrane_bits:
        return 'joint'
    if weight_bits:
        return 'weight-only'
    if membrane_bits:
        return 'membrane-only'
    return 'fp32-control'


def config_tag(weight_bits, membrane_bits, input_size, seed):

    w = weight_bits if weight_bits else 'x'
    u = membrane_bits if membrane_bits else 'x'
    return f'qat_w{w}u{u}_{input_size}px_seed{seed}'


def fp32_checkpoint_for(input_size, seed):
    return RUNS_DIR / f'fp32_{input_size}px_seed{seed}' / 'best_model.pt'


def quantised_state_dict(net):
    # a plain state dict holding the quantised weights
    state = {k: v.detach().clone() for k, v in net.state_dict().items()
             if '.parametrizations.' not in k}
    for name, module in net.named_modules():
        if parametrize.is_parametrized(module, 'weight'):
            state[f'{name}.weight'] = module.weight.detach().clone()
    return state


def run_qat(input_size, seed, hyperparams, weight_bits=None, membrane_bits=None,
            tag=None, n_epochs=N_EPOCHS, patience=PATIENCE, lr_scale=FINETUNE_LR_SCALE,
            device=None, verbose=True, control = False):
    # fine tune one quantised configuration returning its summary
    tag = tag or config_tag(weight_bits, membrane_bits, input_size, seed)
    out_dir = RUNS_DIR / tag
    summary_path = out_dir / 'run.json'

    if summary_path.exists():
        if verbose:
            print(f'[{tag}] already present: skipping')
        with open(summary_path, encoding='utf-8') as f:
            return json.load(f)

    if weight_bits is None and membrane_bits is None and not control:
        raise ValueError('neither target is quantised')

    source = fp32_checkpoint_for(input_size, seed)
    if not source.exists():
        raise FileNotFoundError(
            f'FP32 checkpoint not found: {source}\n')

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / 'best_model.pt'

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset, train_dl, val_dl, _ = build_dataloaders(input_size=input_size,
                                                     verbose=False)

    learning_rate = hyperparams['learning_rate'] * lr_scale
    beta = hyperparams['beta']
    time_steps = hyperparams['time_steps']

    net = GWGlitchSNN(beta=beta, input_size=input_size).to(device)
    net.load_state_dict(torch.load(source, map_location=device))
    criterion = torch.nn.CrossEntropyLoss()

    _, start_accuracy = validate(net, val_dl, criterion, time_steps, device)

    # Attach the quantiser before the optimiser
    handle = apply_quantisation(net, weight_bits=weight_bits,
                                membrane_bits=membrane_bits)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Accuracy with the quantiser attached but no training yet
    _, ptq_accuracy = validate(net, val_dl, criterion, time_steps, device)

    if verbose:
        print(f'\n=== {tag} ===')
        print(f'  {handle.describe()}')
        print(f'  from {source}')
        print(f'  resolution {input_size} px | seed {seed} | T={time_steps} '
              f'| beta={beta:.4f} | lr={learning_rate:.2e} ({lr_scale:g}x)')
        print(f'  FP32 start {start_accuracy:.4f} | after quantisation, before '
              f'fine-tuning {ptq_accuracy:.4f} '
              f'({(ptq_accuracy - start_accuracy) * len(val_dl.dataset):+.0f} samples)')

    best_val_accuracy = ptq_accuracy
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0
    stopped_early = False
    history = []
    epoch = -1

    # The un-fine-tuned model is already a candidat
    torch.save(quantised_state_dict(net), checkpoint)

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
            torch.save(quantised_state_dict(net), checkpoint)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f'  early stopping: {patience} epochs w/o improvement '
                      'of validation loss')
            stopped_early = True
            break

    final_firing_rates = measure_firing_rates(net, val_dl, time_steps, device)
    torch.save(quantised_state_dict(net), out_dir / 'final_model.pt')

    quantisation = handle.config()
    weight_steps = handle.weight_steps() if weight_bits else {}
    handle.remove()

    # per-class recall on the best checkpoint

    net.load_state_dict(torch.load(checkpoint, map_location=device))
    with apply_quantisation(net, membrane_bits=membrane_bits):
        recall, support = validate_per_class(net, val_dl, time_steps, device,
                                             n_classes=len(dataset.classes))
        checkpoint_firing_rates = measure_firing_rates(net, val_dl, time_steps, device)
    per_class = {name: r for name, r in zip(dataset.classes, recall)}

    recovered = best_val_accuracy - ptq_accuracy
    if verbose:
        print(f'  best val accuracy {best_val_accuracy:.4f} (epoch {best_epoch})')
        print(f'  fine-tuning recovered {recovered * len(val_dl.dataset):+.0f} '
              f'samples over PTQ; {(best_val_accuracy - start_accuracy) * len(val_dl.dataset):+.0f} '
              'against the FP32 start')
        for name, r in per_class.items():
            print(f'    {name:<18} recall {r:.4f}')

    summary = {
        'tag': tag,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'mode': 'quantisation-aware fine-tuning',
        'arm': arm_of(weight_bits, membrane_bits),
        'quantisation': quantisation,
        'weight_steps': weight_steps,
        'membrane_range': list(MEMBRANE_RANGE),
        'source_checkpoint': str(source),
        'input_size': input_size,
        'seed': seed,
        'hyperparameters': hyperparams,
        'learning_rate': learning_rate,
        'lr_scale': lr_scale,
        'n_train': len(train_dl.dataset),
        'n_val': len(val_dl.dataset),
        'fp32_start_accuracy': start_accuracy,
        'ptq_accuracy': ptq_accuracy,
        'best_epoch': best_epoch,
        'best_val_accuracy': best_val_accuracy,
        'best_val_loss': best_val_loss,
        'recovered_over_ptq': recovered,
        'val_recall_per_class': per_class,
        'val_support_per_class': dict(zip(dataset.classes, support)),
        'checkpoint_firing_rates': checkpoint_firing_rates,
        'final_firing_rates': final_firing_rates,
        'final_checkpoint': str(out_dir / 'final_model.pt'),
        'epochs_run': epoch + 1,
        'stopped_early': stopped_early,
        'checkpoint': str(checkpoint),
        'history': history,
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    del net, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Quantisation-aware fine-tuning from an FP32 checkpoint')
    parser.add_argument('--weight-bits', type=int, default=None,
                        help='None leaves the weights in FP32 (membrane-only arm)')
    parser.add_argument('--membrane-bits', type=int, default=None,
                        help='None leaves the membrane in FP32 (weight-only arm)')
    parser.add_argument('--input-size', type=int, default=INPUT_SIZE)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--lr-scale', type=float, default=FINETUNE_LR_SCALE)
    parser.add_argument('--control', action='store_true',
                        help='FP32 control arm: identical fine-tuning, quantiser ')
    args = parser.parse_args()

    summary = run_qat(args.input_size, args.seed, load_hyperparams(),
                      weight_bits=args.weight_bits,
                      membrane_bits=args.membrane_bits,
                      tag=args.tag, n_epochs=args.epochs, lr_scale=args.lr_scale,
                      control=args.control)
    
    print(f"\nSummary in {RUNS_DIR / summary['tag'] / 'run.json'}")


if __name__ == '__main__':
    main()