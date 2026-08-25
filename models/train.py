'''
Final training with the best hyperparameters found by Optuna.
'''

import json
import os
import sys
from datetime import datetime

import torch

sys.path.append(os.path.abspath('C:/Users/devam/OneDrive/Tesi'))
from models.snn_model import GWGlitchSNN
from src_python.dataloader import train_dataloader, val_dataloader
from tune import train_one_epoch, validate

HYPERPARAMS_FILE = 'best_hyperparams.json'
CHECKPOINT = 'best_model.pt'
RUN_LOG = 'final_training_run.json'
SEED = 1234
N_EPOCHS = 50
PATIENCE = 5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if os.path.exists(CHECKPOINT):
        backup = f'{os.path.splitext(CHECKPOINT)[0]}_pre_leakage_fix.pt'
        if not os.path.exists(backup):
            os.rename(CHECKPOINT, backup)
            print(f'Checkpoint in {backup}')

    with open(HYPERPARAMS_FILE) as f:
        best_params = json.load(f)
    print("Best hyperparameters:", best_params)

    learning_rate = best_params["learning_rate"]
    beta = best_params["beta"]
    time_steps = best_params["time_steps"]

    print(f"Train batches: {len(train_dataloader)} | Val batches: {len(val_dataloader)}")
    print(f"Train samples: {len(train_dataloader.dataset)} | "
          f"Val samples: {len(val_dataloader.dataset)}")

    net = GWGlitchSNN(beta=beta).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    stopped_early = False

    try:
        for epoch in range(N_EPOCHS):
            avg_firing_rate, layer_firing_rates = train_one_epoch(
                net, train_dataloader, optimizer, criterion, time_steps, device
            )
            val_loss, val_accuracy = validate(
                net, val_dataloader, criterion, time_steps, device
            )

            print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}, "
                  f"avg_firing_rate={avg_firing_rate:.4f}")
            print(f"firing rate per layer: {layer_firing_rates}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                torch.save(net.state_dict(), CHECKPOINT)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"No improvement in validation loss for {PATIENCE} "
                      f"consecutive epochs - stopping")
                stopped_early = True
                break

    except torch.cuda.OutOfMemoryError:
        print("OOM during training - interrupting")
        del net, optimizer
        torch.cuda.empty_cache()
        raise

    print(f"Training complete - best validation accuracy: {best_val_accuracy:.4f} "
          f"(epoch {best_epoch})")

    # tracciabilita' della run: serve nel report e per confrontare piu' seed
    with open(RUN_LOG, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'seed': SEED,
            'hyperparameters': best_params,
            'n_train': len(train_dataloader.dataset),
            'n_val': len(val_dataloader.dataset),
            'best_epoch': best_epoch,
            'best_val_accuracy': best_val_accuracy,
            'best_val_loss': best_val_loss,
            'epochs_run': epoch + 1,
            'stopped_early': stopped_early,
            'checkpoint': CHECKPOINT,
        }, f, indent=2)
    print(f"Riepilogo della run salvato in {RUN_LOG}")


if __name__ == "__main__":
    main()