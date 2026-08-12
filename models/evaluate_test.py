'''
Final evaluation of model on test set
'''
import os
import sys
import json

import torch

sys.path.append(os.path.abspath('C:/Users/devam/OneDrive/Tesi'))
from models.snn_model import GWGlitchSNN
from src_python.dataloader import test_dataloader
from tune import validate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("best_hyperparams.json", "r") as f:
        best_params = json.load(f)

    net = GWGlitchSNN(beta=best_params["beta"]).to(device)
    net.load_state_dict(torch.load("best_model.pt", map_location=device))
    net.eval()

    time_steps = best_params["time_steps"]
    criterion = torch.nn.CrossEntropyLoss()

    test_loss, test_accuracy = validate(net, test_dataloader, criterion, time_steps, device)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()