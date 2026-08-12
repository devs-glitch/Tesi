'''
Final training with best hyperparameters found by Optuna
'''
import os
import sys

import optuna
import torch
import json

sys.path.append(os.path.abspath('C:/Users/devam/OneDrive/Tesi'))
from models.snn_model import GWGlitchSNN
from src_python.dataloader import train_dataloader, val_dataloader
from tune import direct_encode, train_one_epoch, validate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    study = optuna.load_study(
        study_name="glitch_snn_hpo_v4",
        storage="sqlite:///optuna_study.db"
    )
    best_params = study.best_trial.params
    print("Best hyperparameters:", best_params)

    with open("best_hyperparams.json", "w") as f:
        json.dump(best_params, f, ensure_ascii=False)     

    learning_rate = best_params["learning_rate"]
    beta = best_params["beta"]
    time_steps = best_params["time_steps"]

    # reconstruct model, optimizer, criterion
    net = GWGlitchSNN(beta=beta).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    n_epochs_final = 50

    best_val_accuracy = 0.0

    try:
        best_val_loss = float("inf")
        patience = 5          
        patience_counter = 0

        for epoch in range(n_epochs_final):
            avg_firing_rate, layer_firing_rates = train_one_epoch(
                net, train_dataloader, optimizer, criterion, time_steps, device
            )
            val_loss, val_accuracy = validate(net, val_dataloader, criterion, time_steps, device)

            print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}, "
                f"avg_firing_rate={avg_firing_rate:.4f}")
            print(f"firing rate per layer: {layer_firing_rates}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(net.state_dict(), "best_model.pt")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"No improvement in validation loss for {patience} consecutive epochs - interromping")
                break

    except torch.cuda.OutOfMemoryError:
        print("OOM during training - interrupting")
        del net, optimizer
        torch.cuda.empty_cache()

    print(f"Training complete - best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()