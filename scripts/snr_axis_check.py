'''
Hypothesis: The divergence on the XAI explanation is greater on weak glitches than on stronger ones.

'''

import csv
import os
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

DATA_DIR = Path('data/processed')
METADATA_CSV = DATA_DIR / 'selected_dataset.csv'
SPLIT_FILE = Path('data/split_assignment.csv')
CROP = (470, 550)          # (altezza, larghezza), come nel dataloader
N_PER_CLASS = 200
SEED = 0


def center_crop(img, crop=CROP):
    h, w = crop
    W, H = img.size
    left, top = (W - w) // 2, (H - h) // 2
    return img.crop((left, top, left + w, top + h))


def image_features(path):
    '''Misure di intensita' su cio' che la rete effettivamente riceve.

    Gli spettrogrammi Gravity Spy usano una colormap tipo jet: il blu e' energia
    bassa, il rosso alta. La luminanza in scala di grigi non e' monotona lungo
    quella colormap, quindi la misura piu' informativa e' la frazione di pixel
    "caldi" (rosso alto e blu basso), che approssima la frazione di piano
    tempo-frequenza con energia elevata.
    '''
    with Image.open(path) as im:
        arr = np.asarray(center_crop(im.convert('RGB')), dtype=np.float32) / 255.0

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    hot = (r > 0.5) & (b < 0.4)          # zona rossa/gialla della colormap
    very_hot = (r > 0.8) & (g < 0.4) & (b < 0.3)

    return {
        'mean_intensity': float(arr.mean()),
        'std_intensity': float(arr.std()),
        'mean_red': float(r.mean()),
        'mean_blue': float(b.mean()),
        'hot_fraction': float(hot.mean()),
        'very_hot_fraction': float(very_hot.mean()),
        'p99_red': float(np.percentile(r, 99)),
    }


def spearman(x, y):
    '''Correlazione di rango, senza scipy.'''
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float('nan')


def row_index(path):
    name = re.split(r'[\\/]', str(path))[-1]
    return int(re.fullmatch(r'glitch_(\d+)\.png', name).group(1))


def main():
    random.seed(SEED)

    meta = list(csv.DictReader(open(METADATA_CSV, newline='', encoding='utf-8')))

    train_paths = defaultdict(list)
    for row in csv.DictReader(open(SPLIT_FILE, newline='', encoding='utf-8')):
        if row['split'] == 'train':
            train_paths[row['class']].append(row['path'])

    samples = defaultdict(list)
    for cls, paths in train_paths.items():
        for rel in random.sample(paths, min(N_PER_CLASS, len(paths))):
            full = DATA_DIR / rel
            if not full.exists():
                continue
            snr = float(meta[row_index(rel)]['snr'])
            samples[cls].append((snr, image_features(full)))

    features = ['mean_intensity', 'std_intensity', 'mean_red', 'mean_blue',
                'hot_fraction', 'very_hot_fraction', 'p99_red']

    print(f'campioni analizzati: ' +
          ', '.join(f'{c} {len(v)}' for c, v in sorted(samples.items())))
    print('\nSpearman fra log(SNR) e ciascuna misura, DENTRO ogni classe')
    print(f'{"misura":<22}' + ''.join(f'{c[:14]:>16}' for c in sorted(samples)))
    print('-' * (22 + 16 * len(samples)))

    for f in features:
        line = f'{f:<22}'
        for cls in sorted(samples):
            snrs = [np.log(s) for s, _ in samples[cls]]
            vals = [d[f] for _, d in samples[cls]]
            line += f'{spearman(snrs, vals):>16.3f}'
        print(line)

    # la scala di colore e' fissa o per immagine?
    print('\nSatura la colormap? p99 del canale rosso per terzile di SNR, per classe')
    print(f'{"classe":<20}{"SNR basso":>14}{"SNR medio":>14}{"SNR alto":>14}')
    for cls in sorted(samples):
        ordered = sorted(samples[cls], key=lambda t: t[0])
        n = len(ordered) // 3
        thirds = [ordered[:n], ordered[n:2 * n], ordered[2 * n:]]
        line = f'{cls:<20}'
        for t in thirds:
            line += f'{statistics.mean(d["p99_red"] for _, d in t):>14.3f}'
        print(line)

    print('\nCome leggerlo:')
    print('  |rho| > 0.4 dentro classe su hot_fraction  -> l\'SNR si traduce in')
    print('     intensita\': l\'asse SNR e\' praticabile come previsto.')
    print('  |rho| ~ 0 e p99_red costante fra i terzili  -> immagini autoscalate:')
    print('     l\'SNR NON governa la corrente in ingresso. In quel caso l\'ipotesi')
    print('     non muore, cambia asse: stratifica su hot_fraction, che misura')
    print('     direttamente la grandezza che il meccanismo chiama in causa.')


if __name__ == '__main__':
    main()