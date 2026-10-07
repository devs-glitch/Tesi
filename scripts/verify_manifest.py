'''
Verify that the audit trail is externally verifiable
'''
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

MANIFEST = Path('config/baseline_manifest.json')
LOST = Path('data/corrupted_files.txt')
CLASSES = ('Blip', 'Extremely_Loud', 'Scattered_Light', 'Violin_Mode')


def normalise(p):
    # the manifest and the lost-file list were written on Windows
    return str(p).replace('\\', '/')


def key(p):
    # last two components: <class>/<file>, the form used in the split file
    parts = normalise(p).split('/')
    return '/'.join(parts[-2:])


def main():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    split_file = Path(normalise(manifest['split_file']))
    rows = list(csv.DictReader(split_file.open(encoding='utf-8')))
    by_split = Counter(r['split'] for r in rows)
    lost = {key(l) for l in LOST.read_text(encoding='utf-8').split('\n') if l.strip()}
    train_rows = {key(r['path']) for r in rows if r['split'] == 'train'}

    checks = [
        ('digest of the split file',
         hashlib.sha256(split_file.read_bytes()).hexdigest()[:16]
         == manifest['split_sha256_16'],
         f'declared {manifest["split_sha256_16"]}'),

        ('no file appears in two partitions',
         len({r['path'] for r in rows}) == len(rows),
         f'{len(rows)} rows, {len({r["path"] for r in rows})} distinct paths'),

        ('validation size matches the manifest',
         by_split['val'] == manifest['n_val'],
         f'{by_split["val"]} vs {manifest["n_val"]}'),

        ('training size matches once the unreadable files are removed',
         by_split['train'] - len(lost) == manifest['n_train'],
         f'{by_split["train"]} - {len(lost)} vs {manifest["n_train"]}'),

        ('every unreadable file belongs to the training partition',
         lost <= train_rows,
         f'{len(lost)} files, outside training: {sorted(lost - train_rows) or "none"}'),

        ('every partition contains all four classes',
         all({r['class'] for r in rows if r['split'] == s} == set(CLASSES)
             for s in by_split),
         ', '.join(f'{s} {len({r["class"] for r in rows if r["split"] == s})}'
                   for s in sorted(by_split))),

        ('the class column agrees with the directory in the path',
         all(r['path'].replace('\\', '/').split('/')[-2] == r['class'] for r in rows),
         f'{sum(1 for r in rows if r["path"].replace(chr(92), "/").split("/")[-2] != r["class"])} mismatches'),
    ]

    width = max(len(name) for name, *_ in checks)
    failed = 0
    for name, passed, detail in checks:
        failed += not passed
        print(f'{"PASS" if passed else "FAIL"}  {name:<{width}}  {detail}')

    # The checkpoints are not distributed with the repository

    missing = [c for c in manifest['checkpoints'] if not Path(normalise(c)).exists()]
    print(f'\n{len(manifest["checkpoints"]) - len(missing)}/{len(manifest["checkpoints"])} '
          'declared checkpoints present'
          + (' (runs/ is not distributed with the repository)' if missing else ''))

    print(f'\n{by_split["train"]} train / {by_split["val"]} val / {by_split["test"]} test'
          f'  |  {len(rows)} listed, {len(rows) - len(lost)} readable')
    if failed:
        print(f'\n{failed} check(s) failed: the manifest does not describe this partition')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())