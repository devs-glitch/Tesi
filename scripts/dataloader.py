'''
Takes clean data and transforms them into tensors
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

# INPUT_SIZE = int(os.environ.get('SNN_INPUT_SIZE', 224))
DEFAULT_INPUT_SIZE = 112
CROP = (470, 550)

def build_transforms(input_size=DEFAULT_INPUT_SIZE):
    return transforms.Compose([
        transforms.CenterCrop(CROP),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
 
 
tranformations = build_transforms()


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

def build_dataloaders(input_size=DEFAULT_INPUT_SIZE, batch_size=BATCH_SIZE, verbose=True):
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=build_transforms(input_size))
    train_ds, val_ds, test_ds = load_frozen_split(dataset, verbose=verbose)
    return (
        dataset,
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )

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
