'''Takes clean data and transforms them into tensors
'''

import csv
import os
from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_DIR = 'data/processed/'
SPLIT_FILE = Path('data/split_assignment.csv')
BATCH_SIZE = 32
SPLIT_NAMES = ('train', 'val', 'test')

tranformations = transforms.Compose([
    transforms.CenterCrop((470, 550)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_frozen_split(dataset, split_file=SPLIT_FILE, verbose=True):
    '''
    Returns (train, val, test) as subset, according to the froze CSV
    Enumerated files not present are skipped and reported
    '''
    if not split_file.exists():
        raise FileNotFoundError(
            f'{split_file} not extists')

    index_of = {os.path.relpath(p, dataset.root).replace(os.sep, '/'): i
                for i, (p, _) in enumerate(dataset.samples)}

    buckets = {name: [] for name in SPLIT_NAMES}
    missing = []
    with open(split_file, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            rel, split = row['path'], row['split']
            if split not in buckets:
                raise ValueError(f'split unknown in CSV: {split!r}')
            idx = index_of.get(rel)
            if idx is None:
                missing.append(rel)
            else:
                buckets[split].append(idx)

    if verbose and missing:
        print(f'{len(missing)} file in CSV not in disk -> excluding them')
        for rel in missing:
            print(f'  - {rel}')

    return tuple(Subset(dataset, sorted(buckets[name])) for name in SPLIT_NAMES)


def split_report(dataset, subsets):
    lines = []
    header = f'{"class":<20}' + ''.join(f'{n:>10}' for n in SPLIT_NAMES) + f'{"tot":>10}'
    lines += [header, '-' * len(header)]
    per_split = [Counter(dataset.samples[i][1] for i in s.indices) for s in subsets]
    for target, class_name in enumerate(dataset.classes):
        counts = [c.get(target, 0) for c in per_split]
        lines.append(f'{class_name:<20}' + ''.join(f'{c:>10}' for c in counts)
                     + f'{sum(counts):>10}')
    totals = [sum(c.values()) for c in per_split]
    lines += ['-' * len(header),
              f'{"total":<20}' + ''.join(f'{t:>10}' for t in totals) + f'{sum(totals):>10}']
    return '\n'.join(lines)


my_dataset = datasets.ImageFolder(root=DATA_DIR, transform=tranformations)
train_ds, val_ds, test_ds = load_frozen_split(my_dataset)

total_size = len(my_dataset)
train_size, val_size, test_size = len(train_ds), len(val_ds), len(test_ds)

train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

if __name__ == '__main__':
    print(f"Total # images: {total_size}")
    print(f"Train: {train_size} | Validation: {val_size} | Test: {test_size}")
    print()
    print(split_report(my_dataset, (train_ds, val_ds, test_ds)))

    images, tags = next(iter(train_dataloader))

    print(f"\nDimension batch images: {images.shape}")
    print(f"Dimension batch tags: {tags.shape}")