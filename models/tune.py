'''
Hyperparameter optimization with Optuna
'''
import os
import sys

import optuna
import wandb

import torch

sys.path.append(os.path.abspath('C:/Users/devam/OneDrive/Tesi'))
from models.snn_model import GWGlitchSNN
from src_python.dataloader import train_dataloader, val_dataloader


def direct_encode(x, time_steps):
    x = x.unsqueeze(0)
    x = x.repeat(time_steps, 1, 1, 1, 1)
    return x

def train_one_epoch(net, train_dataloader, optimizer, criterion, time_steps, device):
    net.train()

    total_firing_rate = 0
    layer_firing_rates = {"layer1": 0, "layer2": 0, "layer3": 0, "layer4": 0}
    n_batches = 0

    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        encoded_x = direct_encode(images, time_steps=time_steps)

        optimizer.zero_grad()

        spike_out, spike1, spike2, spike3, spike4 = net(encoded_x)
        spike_count = spike_out.sum(dim = 0)

        loss = criterion(spike_count, labels)

        loss.backward()
        optimizer.step()

        total_firing_rate += spike_out.mean().item()
        layer_firing_rates["layer1"] += spike1.mean().item()
        layer_firing_rates["layer2"] += spike2.mean().item()
        layer_firing_rates["layer3"] += spike3.mean().item()
        layer_firing_rates["layer4"] += spike4.mean().item()
        n_batches += 1

    avg_firing_rate = total_firing_rate / n_batches
    avg_layer_firing_rates = {k: v / n_batches for k, v in layer_firing_rates.items()}

    return avg_firing_rate, avg_layer_firing_rates

def validate(net, val_dataloader, criterion, time_steps, device):
    net.eval()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in val_dataloader:
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)

            encoded_x = direct_encode(images, time_steps=time_steps)

            spike_out, _, _, _, _ = net(encoded_x)
            spike_count = spike_out.sum(dim = 0)

            predicted = torch.argmax(spike_count, dim=1)   # classe with more spike for each image of the batch
            correct += (predicted == labels).sum().item()  # correct in this batch
            total += labels.size(0)                        # #images in this batch      

            loss = criterion(spike_count, labels)
            total_loss += loss.item() * labels.size(0)

    return total_loss / total, correct / total

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    beta = trial.suggest_float("beta", 0, 1)
    time_steps = trial.suggest_int("time_steps", 5, 15)

    wandb.init(
        project="Tesi",
        name=f"trial_{trial.number}",
        config={
            "learning_rate": learning_rate,
            "beta": beta,
            "time_steps": time_steps,
        },
        reinit="finish_previous"
    )
    
    net = GWGlitchSNN(beta=beta).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    n_epochs = 15
    val_accuracy = 0.0

    try:
        for epoch in range(n_epochs):
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

    wandb.finish()
    return val_accuracy

if __name__ == "__main__":
    study = optuna.create_study(
    study_name="glitch_snn_hpo_v4",
    storage="sqlite:///optuna_study.db",
    direction="maximize",
    pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=15, reduction_factor=3),
    load_if_exists=True    
)

    study.optimize(objective, n_trials=30)

    print("Best trial:", study.best_trial.params)