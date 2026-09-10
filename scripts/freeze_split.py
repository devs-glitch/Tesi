'''
Freezes the split train/val/test
'''

import os
from pathlib import Path

import torch
from torch.utils.data import random_split
from torchvision import datasets

DATA_DIR = 'data/processed/'
SPLIT_FILE = Path('data/split_assignment.csv')
SEED = 42
SPLIT_NAMES = ('train', 'val', 'test')
EXPECTED_TOTAL = 9568

def main():
    if SPLIT_FILE.exists():
        raise SystemExit(
            f'{SPLIT_FILE} already exists')

    dataset = datasets.ImageFolder(root=DATA_DIR)
    total = len(dataset)
    print(f'Enumerated files: {total}')

    if total != EXPECTED_TOTAL:
        print(f'Expected {EXPECTED_TOTAL} file, found {total}')

    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    generator = torch.Generator().manual_seed(SEED)
    subsets = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_FILE, 'w', encoding='utf-8') as fh:
        fh.write('path,class,split\n')
        for name, subset in zip(SPLIT_NAMES, subsets):
            for i in sorted(subset.indices):
                path, target = dataset.samples[i]
                rel = os.path.relpath(path, dataset.root).replace(os.sep, '/')
                fh.write(f'{rel},{dataset.classes[target]},{name}\n')

    print(f'Train: {train_size} | Validation: {val_size} | Test: {test_size}')
    print(f'Partition freezed at {SPLIT_FILE}')


if __name__ == '__main__':
    main()