'''
Read the Gravity Spy metadata, select classes and features, balance the classes
and draw the exploratory figures of Section 3.1

Run from the repository root:  python -m scripts.data_prep
Inputs:  data/raw/{H1,L1}_{O1,O2,O3a,O3b}.csv  (scripts.dataset_downloader)
Outputs: data/processed/selected_dataset.csv and, in results/Figures/,
         label_counts.png, feature_boxplots.png, correlation_matrix.png,
         pairplot.png, umap.png (the file names used by the thesis)

selected_dataset.csv fixes which spectrogram every glitch_<row>.png is, so it is
never overwritten silently: if it already exists, the new selection must be
identical to it, otherwise the script stops (use --force to replace it).
'''

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler

RAW_DIR = Path('data/raw')
OUT_CSV = Path('data/processed/selected_dataset.csv')
FIG_DIR = Path('results/Figures')

FILE_NAMES = ['H1_O1', 'H1_O2', 'H1_O3a', 'H1_O3b',
              'L1_O1', 'L1_O2', 'L1_O3a', 'L1_O3b']
SELECTED_CLASSES = ['Scattered_Light', 'Blip', 'Extremely_Loud', 'Violin_Mode']
NUMERIC_COLUMNS = ['event_time', 'peak_time', 'peak_time_ns', 'start_time',
                   'start_time_ns', 'duration', 'peak_frequency', 'central_freq',
                   'amplitude', 'snr', 'bandwidth', 'q_value']
COLUMNS_TO_KEEP = [
    'gravityspy_id',                                            # measurement ID
    'duration', 'peak_frequency', 'snr', 'bandwidth',           # numeric features
    'ml_label',                                                 # label
    'url1',                                                     # spectrogram
    'Scattered_Light', 'Blip', 'Extremely_Loud', 'Violin_Mode'  # confidence scores
]

# one colour per class, the same in every figure
_SET1 = sns.color_palette('Set1', 4)
CLASS_COLOURS = {'Blip': _SET1[0], 'Extremely_Loud': _SET1[1],
                 'Scattered_Light': _SET1[2], 'Violin_Mode': _SET1[3]}


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')


def load_release():
    # the eight CSVs of the release, concatenated in a fixed order
    frames = []
    for name in FILE_NAMES:
        frame = pd.read_csv(RAW_DIR / f'{name}.csv')
        frame['detector_run'] = name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_label_counts(df):
    counts = df['ml_label'].value_counts()
    print('class distribution\n', counts)
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
    ax.set_title('Samples per label')
    ax.set_xlabel('label')
    ax.set_ylabel('number of samples')
    ax.tick_params(axis='x', labelrotation=45)
    plt.setp(ax.get_xticklabels(), ha='right')
    fig.tight_layout()
    save(fig, 'label_counts.png')


def plot_boxplots(df):
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 16))
    for ax, col in zip(axes.flatten(), NUMERIC_COLUMNS):
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(col, fontsize=12, pad=20)
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelsize=9)
        ax.yaxis.get_offset_text().set_fontsize(8)
        ax.yaxis.get_offset_text().set_y(1.02)
    fig.tight_layout(pad=3.0, h_pad=4.0, w_pad=2.5)
    save(fig, 'feature_boxplots.png')


def plot_correlation(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[NUMERIC_COLUMNS].corr(), annot=True, cmap='coolwarm',
                fmt='.2f', ax=ax)
    ax.set_title('Feature correlation matrix')
    fig.tight_layout()
    save(fig, 'correlation_matrix.png')


def plot_pairplot(df):
    features = ['duration', 'peak_frequency', 'snr', 'bandwidth', 'ml_label']
    data = df[df['ml_label'].isin(SELECTED_CLASSES)][features].copy()
    data['snr'] = np.log1p(data['snr'])
    data['duration'] = np.log1p(data['duration'])

    g = sns.pairplot(data=data, hue='ml_label', palette=CLASS_COLOURS,
                     hue_order=list(CLASS_COLOURS), corner=True, height=3.5,
                     plot_kws={'alpha': 0.5, 's': 15, 'linewidth': 0},
                     diag_kws={'fill': True, 'alpha': 0.3})
    for ax in g.axes.flatten():
        if ax is None:
            continue
        if ax.get_xlabel() in ('snr', 'duration'):
            ax.set_xlabel(f'log({ax.get_xlabel()} + 1)')
        if ax.get_ylabel() in ('snr', 'duration'):
            ax.set_ylabel(f'log({ax.get_ylabel()} + 1)')
    sns.move_legend(g, 'center right', bbox_to_anchor=(1.02, 0.5),
                    title='Class', fontsize=10)
    g.figure.suptitle('Duration, peak frequency, SNR and bandwidth by class',
                      y=1.02, fontsize=16)
    save(g.figure, 'pairplot.png')


def select(df):
    selected = df.loc[df['ml_label'].isin(SELECTED_CLASSES),
                      COLUMNS_TO_KEEP].reset_index(drop=True)
    print(f'selected: {selected.shape}\n', selected['ml_label'].value_counts())

    # events without a spectrogram address
    selected = selected.loc[selected['url1'] != '?'].reset_index(drop=True)

    # balance to the smallest class
    n_min = selected['ml_label'].value_counts().min()
    return (selected.groupby('ml_label').sample(n=n_min, random_state=42)
            .reset_index(drop=True))


def plot_umap(selected):
    sample = selected.groupby('ml_label').sample(n=500, random_state=42)
    X = StandardScaler().fit_transform(
        sample[['duration', 'peak_frequency', 'snr', 'bandwidth']])
    embedding = umap.UMAP(n_components=2, n_neighbors=15,
                          random_state=42).fit_transform(X)
    frame = pd.DataFrame({'UMAP 1': embedding[:, 0], 'UMAP 2': embedding[:, 1],
                          'Class': sample['ml_label'].values})
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=frame, x='UMAP 1', y='UMAP 2', hue='Class',
                    hue_order=list(CLASS_COLOURS), palette=CLASS_COLOURS,
                    s=20, alpha=0.8, ax=ax)
    ax.set_title('UMAP projection of the four selected classes', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    save(fig, 'umap.png')


def write_selection(selected, force=False):
    if OUT_CSV.exists() and not force:
        existing = pd.read_csv(OUT_CSV)
        same = (len(existing) == len(selected)
                and existing['gravityspy_id'].astype(str).tolist()
                == selected['gravityspy_id'].astype(str).tolist())
        if not same:
            raise SystemExit(
                f'{OUT_CSV} exists and the new selection differs from it: the '
                'images glitch_<row>.png would no longer match their rows. '
                'Nothing was written; use --force only to rebuild the corpus.')
        print(f'{OUT_CSV} already exists and is identical: left untouched')
        return
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUT_CSV, index=False)
    print(f'written {OUT_CSV}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true',
                        help='overwrite selected_dataset.csv even if it differs')
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()

    df = load_release()
    print(f'release: {df.shape}')

    if not args.no_figures:
        print('figures:')
        plot_label_counts(df)
        plot_boxplots(df)
        plot_correlation(df)
        plot_pairplot(df)

    selected = select(df)
    if not args.no_figures:
        plot_umap(selected)
    write_selection(selected, force=args.force)


if __name__ == '__main__':
    main()
