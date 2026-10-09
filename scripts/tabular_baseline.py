'''
Tabular baseline on the four classes
'''
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

METADATA = Path('data/processed/selected_dataset.csv')
SPLIT_FILE = Path('data/split_assignment.csv')
LOST = Path('data/corrupted_files.txt')
GRID_DIR = Path('results/grid')
OUT_JSON = Path('results/tabular_baseline.json')
OUT_CSV = Path('results/tables/tabular_baseline.csv')

FEATURES = ('duration', 'peak_frequency', 'snr', 'bandwidth')

LOGGED = ('snr', 'duration')

SEEDS = (1234, 2345, 3456)


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return float('nan'), float('nan')
    p = successes / total
    denom = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denom
    half = z * ((p * (1 - p) / total + z ** 2 / (4 * total ** 2)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def build_table():
    # Join the metadata to the frozen split, exactly as in stratify.py
    with open(METADATA, newline='', encoding='utf-8') as f:
        metadata = list(csv.DictReader(f))
    lost = {line.strip().replace('\\', '/').split('/')[-1]
            for line in LOST.read_text(encoding='utf-8').splitlines() if line.strip()}

    rows = []
    with open(SPLIT_FILE, newline='', encoding='utf-8') as f:
        for record in csv.DictReader(f):
            # validation only
            if record['split'] == 'test':
                continue
            path = record['path'].replace('\\', '/')
            folder, filename = path.split('/')[-2:]
            if filename in lost:
                continue
            match = re.fullmatch(r'glitch_(\d+)\.png', filename)
            if match is None:
                raise SystemExit(f'unexpected file name: {filename}')
            index = int(match.group(1))
            if index >= len(metadata):
                raise SystemExit(
                    f'{filename} points at row {index}, past the end of the '
                    f'metadata ({len(metadata)} rows): the CSV is not the one '
                    'these images were downloaded from')
            meta = metadata[index]
            if meta['ml_label'] != folder:
                raise SystemExit(
                    f'join check failed on {path}: row {index} of the metadata '
                    f'is {meta["ml_label"]}, the file is under {folder}')
            rows.append({'split': record['split'], 'class': folder, 'meta': meta})
    return rows, metadata[0].keys()


def feature_matrix(rows, columns):
    x = []
    for record in rows:
        values = []
        for name in columns:
            raw = record['meta'][name]
            try:
                v = float(raw)
            except (TypeError, ValueError):
                raise SystemExit(
                    f'{name} is {raw!r} on a {record["class"]} row of the '
                    'metadata: every retained descriptor has to be numeric, so '
                    'the table needs cleaning before the baseline can run')
            values.append(v)
            if name in LOGGED:
                values.append(np.log1p(max(v, 0.0)))
        x.append(values)
    return np.asarray(x, dtype=np.float64)


def snn_reference():
    #network's own validation accuracy
    accuracies = []
    for path in sorted(GRID_DIR.glob('fp32_*.json')):
        with open(path, encoding='utf-8') as f:
            accuracies.append(json.load(f)['accuracy'])
    return accuracies


def main():
    parser = argparse.ArgumentParser(
        description='Tabular baseline on the four retained Omicron descriptors, '
                    'against the network, on the same validation images')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if OUT_JSON.exists() and not args.force:
        raise SystemExit(f'{OUT_JSON} already exists. Pass --force to replace it')

    from sklearn.base import clone
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows, available = build_table()
    missing = [c for c in FEATURES if c not in available]
    if missing:
        raise SystemExit(f'{METADATA} has no column {missing}: it is not the '
                         'table the feature selection produced')
    columns = list(FEATURES)

    classes = sorted({r['class'] for r in rows})
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'val']
    x_train, x_val = feature_matrix(train, columns), feature_matrix(val, columns)
    y_train = np.array([classes.index(r['class']) for r in train])
    y_val = np.array([classes.index(r['class']) for r in val])

    names = []
    for c in columns:
        names.append(c)
        if c in LOGGED:
            names.append(f'log1p({c})')

    print('=' * 72)
    print('TABULAR BASELINE  |  Omicron descriptors, no spectrogram')
    print(f'{len(train)} training rows, {len(val)} validation rows, '
          f'{len(classes)} classes')
    print(f'features ({len(names)}): ' + ', '.join(names))
    print('=' * 72)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    definitions = {
        'majority class': (DummyClassifier(strategy='most_frequent'), {}),
        'logistic regression': (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
            {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}),
        'random forest': (
            RandomForestClassifier(n_estimators=400, n_jobs=-1),
            {'max_depth': [None, 12], 'min_samples_leaf': [1, 3]}),
        'gradient boosting': (
            HistGradientBoostingClassifier(),
            {'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [31, 63]}),
    }

    results, table = {}, []
    header = (f'\n{"model":<22}{"seeds":>6}{"accuracy":>10}{"min":>9}{"max":>9}'
              f'{"95% Wilson":>22}{"errors":>9}')
    print(header)
    print('-' * (len(header) - 1))

    for label, (estimator, grid) in definitions.items():
        '''
        Hyperparameters are chosen once, by cross-validation inside the
        training partition; the three seeds then vary only the fit, which is
        what makes the spread comparable with the network's three seeds
        '''
        best_params = None
        if grid:
            search = GridSearchCV(clone(estimator), grid, cv=folds, n_jobs=-1,
                                  scoring='accuracy')
            search.fit(x_train, y_train)
            best_params = search.best_params_
            estimator = search.best_estimator_

        per_seed = []
        for seed in SEEDS:
            model = clone(estimator)
            if 'random_state' in model.get_params():
                model.set_params(random_state=seed)
            else:
                for name in model.get_params():
                    if name.endswith('__random_state'):
                        model.set_params(**{name: seed})
            fitted = model.fit(x_train, y_train)
            predicted = fitted.predict(x_val)
            per_seed.append({
                'accuracy': float((predicted == y_val).mean()),
                'recall': {c: float((predicted[y_val == i] == i).mean())
                           for i, c in enumerate(classes)},
                'confusion': [[int(((y_val == i) & (predicted == j)).sum())
                               for j in range(len(classes))]
                              for i in range(len(classes))],
            })

        accuracy = float(np.mean([s['accuracy'] for s in per_seed]))
        correct = int(round(accuracy * len(y_val)))
        lo, hi = wilson(correct, len(y_val))
        results[label] = {
            'accuracy_mean': accuracy,
            'accuracy_min': min(s['accuracy'] for s in per_seed),
            'accuracy_max': max(s['accuracy'] for s in per_seed),
            'wilson_95': [lo, hi],
            'errors': len(y_val) - correct,
            'best_params': {k: str(v) for k, v in (best_params or {}).items()},
            'recall': {c: float(np.mean([s['recall'][c] for s in per_seed]))
                       for c in classes},
            'confusion_seed1234': per_seed[0]['confusion'],
            'confusion_per_seed': [s['confusion'] for s in per_seed],
        }
        print(f'{label:<22}{len(SEEDS):>6}{accuracy:>10.4f}'
              f'{results[label]["accuracy_min"]:>9.4f}'
              f'{results[label]["accuracy_max"]:>9.4f}'
              f'{f"[{lo:.4f}, {hi:.4f}]":>22}'
              f'{results[label]["errors"]:>9}')
        for c in classes:
            table.append({'model': label, 'class': c,
                          'recall': results[label]['recall'][c],
                          'accuracy_mean': accuracy,
                          'n_val': len(y_val)})

    # the network
    snn = snn_reference()
    best_label = max((k for k in results if k != 'majority class'),
                     key=lambda k: results[k]['accuracy_mean'])
    best = results[best_label]['accuracy_mean']
    snn_mean = float(np.mean(snn)) if snn else float('nan')

    print(f'\n{"spiking network (FP32)":<22}{len(snn):>6}{snn_mean:>10.4f}'
          f'{min(snn):>9.4f}{max(snn):>9.4f}'
          + f'{"":>22}{len(y_val) - int(round(snn_mean * len(y_val))):>9}')

    print(f'\nbest baseline: {best_label}, {best:.4f}')
    if snn:
        baseline_error, snn_error = 1 - best, 1 - snn_mean
        print(f'error rate {baseline_error * 100:.2f} % against '
              f'{snn_error * 100:.2f} %, a factor of '
              f'{baseline_error / snn_error:.1f}')
        print(f'{(best - snn_mean) * len(y_val):+.0f} samples out of {len(y_val)}')

    print('\nper-class recall')
    print(f'{"model":<22}' + ''.join(f'{c[:16]:>18}' for c in classes))
    for label in results:
        print(f'{label:<22}'
              + ''.join(f'{results[label]["recall"][c]:>18.4f}' for c in classes))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'metadata': str(METADATA), 'features': names,
                   'n_train': len(train), 'n_val': len(val),
                   'classes': classes, 'seeds': list(SEEDS),
                   'snn_val_accuracy': snn, 'models': results}, f, indent=2)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f'\nwritten to {OUT_JSON}\n            {OUT_CSV}')


if __name__ == '__main__':
    main()