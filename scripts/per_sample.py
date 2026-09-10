'''
Per sample metrics and dataset's physical metadata

The hypothesis is that explanation divergence between the FP32 model and a quantized one
should be larger for weak glitches than for loud ones.

Membrane-potential quantization applies a fixed absolute step to U.
Weak events have small margins, so they suffer disproportionately -> STEEP SNR dependence.

Weight quantization perturbs the input current roughly multiplicatively:
a strong drive gets a proportionally larger absolute error,
so the margin-to-error ratio stays closer to constant -> FLATTER SNR dependence
'''

import csv
import os
import re
from pathlib import Path

METADATA_CSV = Path('data/processed/selected_dataset.csv')
NUMERIC_COLUMNS = ('snr', 'duration', 'peak_frequency', 'bandwidth')


def loader_paths(dataloader):
    # Path per sample, in the order of the dataloader
    
    subset = dataloader.dataset
    if getattr(dataloader, 'sampler', None) is not None:
        from torch.utils.data import RandomSampler
        if isinstance(dataloader.sampler, RandomSampler):
            raise ValueError(
                'loader_paths needs shuffle=False')
    dataset = subset.dataset
    return [os.path.relpath(dataset.samples[i][0], dataset.root).replace(os.sep, '/')
            for i in subset.indices]


def row_index(path):
    # row index in selected_dataset.csv
    name = re.split(r'[\\/]', str(path))[-1]
    match = re.fullmatch(r'glitch_(\d+)\.png', name)
    if match is None:
        raise ValueError(f'unknown file name: {path}')
    return int(match.group(1))


def save_per_sample(rows, out_path, fieldnames=None):
    # write CSV per sample

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or list(rows[0].keys())
    if 'path' not in fieldnames:
        raise ValueError('path not in fieldnames')
    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} rows per sample in {out_path}')
    return out_path


def load_metadata(metadata_csv=METADATA_CSV):
    # metadata per row index (also with confidence value for the classes of the dataset)
    meta = {}
    with open(metadata_csv, newline='', encoding='utf-8') as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            entry = {c: float(row[c]) for c in NUMERIC_COLUMNS}
            entry['ml_label'] = row['ml_label']
            entry['gravityspy_id'] = row['gravityspy_id']
            entry['label_confidence'] = float(row[row['ml_label']])
            meta[i] = entry
    return meta


def join_metadata(per_sample_csv, metadata_csv=METADATA_CSV, out_path=None):

    # joins physical data to a CSV per sample
    meta = load_metadata(metadata_csv)
    rows = []
    with open(per_sample_csv, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            row.update(meta[row_index(row['path'])])
            rows.append(row)
    if out_path:
        save_per_sample(rows, out_path)
    return rows
