'''
Shared training function btw tune.py, train.y and resolution_study.py
'''

import torch


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
        spike_count = spike_out.sum(dim=0)

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
            spike_count = spike_out.sum(dim=0)

            predicted = torch.argmax(spike_count, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loss = criterion(spike_count, labels)
            total_loss += loss.item() * labels.size(0)

    return total_loss / total, correct / total


def validate_per_class(net, dataloader, time_steps, device, n_classes=4):
    # Recall per class. For resolution study
    net.eval()

    correct = torch.zeros(n_classes, dtype=torch.long)
    support = torch.zeros(n_classes, dtype=torch.long)

    for images, labels in dataloader:
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)

            encoded_x = direct_encode(images, time_steps=time_steps)
            spike_out, _, _, _, _ = net(encoded_x)
            predicted = torch.argmax(spike_out.sum(dim=0), dim=1)

            for c in range(n_classes):
                mask = labels == c
                support[c] += mask.sum().item()
                correct[c] += (predicted[mask] == c).sum().item()

    recall = [(correct[c] / support[c]).item() if support[c] > 0 else float('nan')
              for c in range(n_classes)]
    return recall, support.tolist()