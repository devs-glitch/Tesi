'''
Why the weight scale is computed per output channel
'''

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path('results/Figures')

BITS = 4
TENSOR_MAX = 0.099      # conv4, Table tab:weight-ranges
CHANNEL_MAX = 0.044     # weakest channel of conv4

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d8d7d2'
W_EDGE, W_FILL = '#2a78d6', '#dce9f8'
DEAD = '#b9b8b3'

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': None,
    'figure.facecolor': 'white', 'font.size': 9,
    'text.color': INK, 'axes.labelcolor': INK,
})


def signed_levels(bits):
    return 2 ** (bits - 1) - 1          # model/quantization.py


def draw_row(ax, step, title, note):
    qmax = signed_levels(BITS)
    points = np.arange(-qmax, qmax + 1) * step
    reachable = np.abs(points) <= CHANNEL_MAX + step / 2

    ax.axvspan(-CHANNEL_MAX, CHANNEL_MAX, color=W_FILL, alpha=0.75,
               linewidth=0, zorder=1)
    ax.axhline(0, color=MUTED, linewidth=0.9, zorder=2)

    # A level the weak channel can never land on is drawn as spent capacity
    ax.vlines(points[~reachable], -0.3, 0.3, color=DEAD, linewidth=1.3, zorder=3)
    ax.vlines(points[reachable], -0.3, 0.3, color=W_EDGE, linewidth=1.8, zorder=4)
    ax.plot(points[reachable], np.zeros(reachable.sum()), 'o', color=W_EDGE,
            markersize=4, zorder=5)

    for sign in (-1, 1):
        ax.annotate('', xy=(sign * CHANNEL_MAX, 0.52), xytext=(0, 0.52),
                    arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    ax.annotate(f'range of the weakest channel, $\\pm{CHANNEL_MAX:g}$',
                (0, 0.58), ha='center', va='bottom', fontsize=6.8, color=MUTED)

    ax.set_xlim(-0.112, 0.112)
    ax.set_ylim(-0.85, 0.95)
    ax.set_yticks([])
    for side in ('left', 'right', 'top', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='x', length=0, colors=MUTED, labelsize=7.5)
    ax.set_xticks([-0.099, -0.044, 0, 0.044, 0.099])

    ax.annotate(title, (0.0, 0.95), xycoords=('axes fraction', 'axes fraction'),
                ha='left', va='top', fontsize=8.4, weight='bold')
    ax.annotate(note, (1.0, 0.95), xycoords='axes fraction', ha='right',
                va='top', fontsize=7.2, color=MUTED)
    return int(reachable.sum()), len(points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    args = parser.parse_args()

    qmax = signed_levels(BITS)
    shared, own = TENSOR_MAX / qmax, CHANNEL_MAX / qmax

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.0))
    height = fig.get_figheight()
    hit_a, total = draw_row(
        axes[0], shared, 'One scale for the whole tensor',
        f'step {shared:.5f}, from $\\max|w| = {TENSOR_MAX:g}$')
    hit_b, _ = draw_row(
        axes[1], own, 'One scale per output channel',
        f'step {own:.5f}, from $\\max|w| = {CHANNEL_MAX:g}$')

    axes[0].annotate(f'{hit_a} of {total} levels reachable', (0, -0.55),
                     ha='center', va='top', fontsize=7.6, color=W_EDGE,
                     weight='bold')
    axes[1].annotate(f'{hit_b} of {total} levels reachable', (0, -0.55),
                     ha='center', va='top', fontsize=7.6, color=W_EDGE,
                     weight='bold')
    axes[1].set_xlabel('weight value, conv4 at 4 bits', fontsize=8, labelpad=1)

    fig.suptitle('What the weakest channel can represent', x=0.004,
                 y=1 - 0.05 / height, ha='left', va='top', fontsize=10.5,
                 weight='bold')
    fig.tight_layout()
    fig.subplots_adjust(top=1 - 0.36 / height, hspace=0.42)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ('png', 'pdf'):
        path = args.out_dir / f'weight_scale_granularity.{suffix}'
        fig.savefig(path)
        written.append(path)
    plt.close(fig)

    print(f'{BITS} bit, qmax {qmax}: shared step {shared:.5f}, own {own:.5f}')
    print(f'  weakest channel: {hit_a} of {total} levels under the shared '
          f'scale, {hit_b} of {total} under its own')
    print('\n'.join(str(p) for p in written))


if __name__ == '__main__':
    main()