'''
Assess where the models disagree, and how the firing rate distributions move

'''

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

GRID_DIR = Path('results/grid')
OUT_JSON = Path('results/grid_analysis.json')

FP32_TAG = re.compile(r'^fp32_(\d+)px_seed(\d+)$')
QAT_TAG = re.compile(r'^qat_(\w+)_(\d+)px_seed(\d+)$')

def load_records(input_size):
    # Every evaluate_model.py record at the manifest resolution
    records = {}
    for path in sorted(GRID_DIR.glob('*.json')):
        tag = path.stem
        fp32, qat = FP32_TAG.match(tag), QAT_TAG.match(tag)
        if fp32:
            size = int(fp32.group(1))
        elif qat:
            size = int(qat.group(2))
        else:
            continue
        if size != input_size:
            continue
        with open(path, encoding='utf-8') as f:
            records[tag] = json.load(f)
    return records


def load_predictions(tag):
    #per-sample rows as arrays indexed by the dataset row

    path = GRID_DIR / f'{tag}_per_sample.csv'
    index, label, prediction = [], [], []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            index.append(int(row['index']))
            label.append(int(row['label']))
            prediction.append(int(row['prediction']))
    order = np.argsort(index)
    return (np.array(label)[order], np.array(prediction)[order])


def load_rates(tag):
    with np.load(GRID_DIR / f'{tag}_rates.npz') as data:
        return {key: data[key] for key in data.files}


def emd(a, b):
    # 1-D earth mover's distance between two equally sized samples
    if a.size != b.size:
        raise ValueError(f'neuron counts differ: {a.size} vs {b.size}')
    return float(np.abs(np.sort(a) - np.sort(b)).mean())

def compare_predictions(reference, candidate):
    # agreement + whether the two models fail on the same samples
    ref_labels, ref_pred = reference
    cand_labels, cand_pred = candidate
    if not np.array_equal(ref_labels, cand_labels):
        raise ValueError('the two per-sample files are not the same images')

    ref_wrong = ref_pred != ref_labels
    cand_wrong = cand_pred != cand_labels
    differ = ref_pred != cand_pred

    both = int((ref_wrong & cand_wrong).sum())
    union = int((ref_wrong | cand_wrong).sum())

    return {
        'n': int(ref_labels.size),
        'reference_errors': int(ref_wrong.sum()),
        'candidate_errors': int(cand_wrong.sum()),
        'shared_errors': both,
        'error_jaccard': (both / union) if union else float('nan'),
        'predictions_differ': int(differ.sum()),
        'differ_fraction': float(differ.mean()),
        # where they disagree, who is right
        'reference_right_candidate_wrong': int((differ & ~ref_wrong & cand_wrong).sum()),
        'candidate_right_reference_wrong': int((differ & ref_wrong & ~cand_wrong).sum()),
        'both_wrong_differently': int((differ & ref_wrong & cand_wrong).sum()),
    }


def compare_rates(reference, candidate):
    layers = sorted(set(reference) & set(candidate))
    return {layer: emd(reference[layer], candidate[layer]) for layer in layers}

def main():
    parser = argparse.ArgumentParser(
        description='Disagreement subset and firing-rate EMD, from the grid files')
    parser.add_argument('--manifest', type=str, default='baseline_manifest.json')
    parser.add_argument('--input-size', type=int, default=None,
                        help='default: the manifest resolution')
    args = parser.parse_args()

    input_size = args.input_size
    if input_size is None:
        with open(args.manifest, encoding='utf-8') as f:
            input_size = json.load(f)['input_size']

    records = load_records(input_size)
    if not records:
        raise SystemExit(f'no grid records at {input_size} px under {GRID_DIR}/')

    fp32_of_seed = {int(FP32_TAG.match(t).group(2)): t
                    for t in records if FP32_TAG.match(t)}
    if not fp32_of_seed:
        raise SystemExit('no FP32 record found; the comparisons have no reference')

    predictions = {t: load_predictions(t) for t in records}
    rates = {t: load_rates(t) for t in records}

    # anchors
    seeds = sorted(fp32_of_seed)
    cross_seed = {}
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            cross_seed[f'{a} vs {b}'] = {
                'predictions': compare_predictions(predictions[fp32_of_seed[a]],
                                                   predictions[fp32_of_seed[b]]),
                'emd': compare_rates(rates[fp32_of_seed[a]], rates[fp32_of_seed[b]]),
            }

    output = {'input_size': input_size,
              'independent_solutions': cross_seed,
              'runs': {}}

    print(f'{input_size} px | {len(records)} models | '
          f'{predictions[fp32_of_seed[seeds[0]]][0].size} validation images')

    print('\nAnchor independent solutions: two FP32 models of different seeds')
    print(f'{"pair":<16}{"differ":>9}{"shared err":>12}{"Jaccard":>10}'
          + ''.join(f'{"EMD " + k:>12}' for k in sorted(next(iter(cross_seed.values()))['emd'])))
    for pair, r in cross_seed.items():
        p, e = r['predictions'], r['emd']
        print(f'{pair:<16}{p["predictions_differ"]:>9}{p["shared_errors"]:>12}'
              f'{p["error_jaccard"]:>10.3f}'
              + ''.join(f'{e[k]:>12.4f}' for k in sorted(e)))

    # Rruns
    qat = sorted(t for t in records if QAT_TAG.match(t))
    print('\nEach run against the FP32 model of its own seed')
    header = (f'{"run":<28}{"acc":>8}{"differ":>8}{"R+/C-":>7}{"C+/R-":>7}'
              f'{"shared err":>12}{"Jaccard":>9}'
              + ''.join(f'{"EMD L" + k[-1]:>10}' for k in sorted(rates[qat[0]])))
    print('\n' + header)
    print('-' * len(header))

    for tag in qat:
        seed = int(QAT_TAG.match(tag).group(3))
        reference_tag = fp32_of_seed.get(seed)
        if reference_tag is None:
            print(f'{tag:<28}  no FP32 reference for seed {seed}: skipped')
            continue

        p = compare_predictions(predictions[reference_tag], predictions[tag])
        e = compare_rates(rates[reference_tag], rates[tag])
        output['runs'][tag] = {
            'arm': records[tag].get('arm'),
            'weight_bits': records[tag].get('weight_bits'),
            'membrane_bits': records[tag].get('membrane_bits'),
            'seed': seed,
            'reference': reference_tag,
            'accuracy': records[tag].get('accuracy'),
            'predictions': p,
            'emd': e,
        }
        print(f'{tag:<28}{records[tag].get("accuracy", float("nan")):>8.4f}'
              f'{p["predictions_differ"]:>8}'
              f'{p["reference_right_candidate_wrong"]:>7}'
              f'{p["candidate_right_reference_wrong"]:>7}'
              f'{p["shared_errors"]:>12}{p["error_jaccard"]:>9.3f}'
              + ''.join(f'{e[k]:>10.4f}' for k in sorted(e)))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f'\nwritten to {OUT_JSON}')


if __name__ == '__main__':
    main()