'''
Hyperparameter optimization with Optuna
'''
import os
import sys
import json

import optuna
import wandb

import torch

sys.path.append(os.path.abspath('C:/Users/devam/OneDrive/Tesi'))
from models.snn_model import GWGlitchSNN
from src_python.dataloader import train_dataloader, val_dataloader, INPUT_SIZE
from src_python.training_utils import train_one_epoch, validate

BETA_SHIFT = [0.25, 0.5, 0.75, 0.875]
SEARCH_SPACE = os.environ.get('SEARCH_SPACE', 'v5')
HYPERPARAMS_FILE = 'best_hyperparams.json'
N_TRIALS = 30
N_EPOCHS = 15

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if SEARCH_SPACE == 'v5':
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-3, log = True)
        beta = trial.suggest_categorical("beta", BETA_SHIFT)
        time_steps = trial.suggest_int("time_steps", 5, 8)
    else:
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log = True)
        beta = trial.suggest_float("beta", 0, 1)
        time_steps = trial.suggest_int("time_steps", 5, 15)

    wandb.init(
        project="Tesi",
        name=f"{SEARCH_SPACE}_{INPUT_SIZE}PX_trial_{trial.number}",
        config={
            "learning_rate": learning_rate,
            "beta": beta,
            "time_steps": time_steps,
            "input_size": INPUT_SIZE,
        },
        reinit="finish_previous"
    )
    
    net = GWGlitchSNN(beta=beta, input_size=INPUT_SIZE).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    val_accuracy = 0.0

    try:
        for epoch in range(N_EPOCHS):
            avg_firing_rate, layer_firing_rates = train_one_epoch(
            net, train_dataloader, optimizer, criterion, time_steps, device
            )
            val_loss, val_accuracy = validate(net, val_dataloader, criterion, time_steps, device)

            wandb.log({
                "epoch": epoch,
                "validation_loss": val_loss,
                "validation accuracy": val_accuracy,
                "avg_firing_rate": avg_firing_rate,
                "firing_rate_layer1": layer_firing_rates["layer1"],
                "firing_rate_layer2": layer_firing_rates["layer2"],
                "firing_rate_layer3": layer_firing_rates["layer3"],
                "firing_rate_layer4": layer_firing_rates["layer4"],
            })

            trial.report(val_accuracy, epoch)
            if trial.should_prune():
                wandb.finish()
                raise optuna.TrialPruned()

    except torch.cuda.OutOfMemoryError:
        del net, optimizer
        torch.cuda.empty_cache()
        wandb.finish()
        raise optuna.TrialPruned()

    del net, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    wandb.finish()
    return val_accuracy

if __name__ == "__main__":
    study_name=f"glitch_snn_hpo_{SEARCH_SPACE}_{INPUT_SIZE}px"
    print(f"Study: {study_name} | resolution {INPUT_SIZE} px")
    if SEARCH_SPACE == 'v5':
        print(f"Search space: lr=[1e-5, 3e-3]log | beta {BETA_SHIFT} | T=[5,8]")
    else:
        print("Search space: lr=[1e-4, 3e-3] log | beta [0,1] continuous | T=[5, 15]")


    study = optuna.create_study(
        study_name=study_name,
        storage="sqlite:///optuna_study.db",
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=N_EPOCHS, reduction_factor=3),
        load_if_exists=True
    )

    study.optimize(objective, n_trials=N_TRIALS)

    print("Best trial:", study.best_trial.number)
    print("Best value:", study.best_value)
    print("Best params:", study.best_trial.params)


    out_file = (HYPERPARAMS_FILE if SEARCH_SPACE == 'v5'
                else f'best_hyperparams_{SEARCH_SPACE}.json')
    with open(out_file, "w") as f:
        json.dump({**study.best_trial.params,
                   "input_size": INPUT_SIZE,
                   "search_space": SEARCH_SPACE,
                   "study_name": study_name,
                   "best_val_accuracy": study.best_value}, f, indent=2)
    print(f"Written to {out_file}")

    # Accuracy per beta value
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed and SEARCH_SPACE == 'v5':
        by_beta = {}
        for t in completed:
            b = t.params["beta"]
            by_beta.setdefault(b, []).append(t.value)
        print("\nBest validation accuracy per beta (completed trials only):")
        for b in sorted(by_beta):
            v = by_beta[b]
            print(f"  beta={b:<6} n={len(v):<3} best={max(v):.4f}  mean={sum(v)/len(v):.4f}")