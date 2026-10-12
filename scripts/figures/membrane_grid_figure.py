'''
The membrane grid at each bit-width, drawn as a number line
'''

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path('results/Figures')

MEMBRANE_RANGE = (0.0, 2.0)      # model/quantization.py
THRESHOLD = 1.0
BITS = (8, 4, 2)

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d8d7d2'
U_EDGE, U_FILL = '#eb6834', '#fbe2d6'   # the membrane quantiser's colour

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': None,
    'figure.facecolor': 'white', 'font.size': 9,
    'text.color': INK, 'axes.labelcolor': INK,
})


def levels(bits, value_range=MEMBRANE_RANGE):
    # representable potentials, unsigned branch of model.quantization.grid
    lo, hi = value_range
    qmax = 2 ** bits - 1
    scale = hi / qmax
    return np.arange(0, qmax + 1) * scale + lo, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    args = parser.parse_args()

    fig, axes = plt.subplots(len(BITS), 1, figsize=(7.2, 2.9))
    height = fig.get_figheight()

    for ax, bits in zip(axes, BITS):
        points, scale = levels(bits)
        below = int(((points > 0) & (points < THRESHOLD)).sum())

        # The sub-threshold region is what the 2-bit row is about
        ax.axvspan(0, THRESHOLD, color=U_FILL, alpha=0.45, linewidth=0, zorder=1)
        ax.axhline(0, color=MUTED, linewidth=0.9, zorder=2)

        # A dense row of ticks reads as a ruler; a sparse one as a few states
        lw = 0.35 if bits == 8 else (0.9 if bits == 4 else 1.6)
        ax.vlines(points, -0.28, 0.28, color=INK, linewidth=lw, zorder=3)
        if bits == 2:
            ax.plot(points, np.zeros_like(points), 'o', color=INK,
                    markersize=4.5, zorder=4)
            for p in points:
                ax.annotate(f'{p:.3g}', (p, -0.34), ha='center', va='top',
                            fontsize=7, color=MUTED)

        ax.axvline(THRESHOLD, color=U_EDGE, linestyle=(0, (5, 2)),
                   linewidth=1.3, zorder=5)

        ax.set_xlim(-0.03, 2.03)
        ax.set_ylim(-0.75, 0.75)
        ax.set_yticks([])
        for side in ('left', 'right', 'top', 'bottom'):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis='x', length=0, labelbottom=(bits == BITS[-1]),
                       colors=MUTED, labelsize=7.5)
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])

        ax.annotate(f'{bits} bit', (-0.02, 0.0), xycoords=('axes fraction', 'data'),
                    ha='right', va='center', fontsize=8.5, weight='bold')
        ax.annotate(f'{len(points)} levels, step {scale:.4f}\n'
                    f'{below} below threshold',
                    (1.012, 0.0), xycoords=('axes fraction', 'data'),
                    ha='left', va='center', fontsize=7.2, color=MUTED,
                    linespacing=1.5)

    axes[-1].set_xlabel('membrane potential, in units of the firing threshold',
                        fontsize=8, labelpad=2)
    axes[0].annotate('threshold', (THRESHOLD, 0.62), ha='center', va='bottom',
                     fontsize=7.5, color=U_EDGE, weight='bold')

    fig.suptitle('The membrane grid over the clipped range $[0, 2]$',
                 x=0.004, y=1 - 0.05 / height, ha='left', va='top',
                 fontsize=10.5, weight='bold')
    fig.tight_layout()
    fig.subplots_adjust(top=1 - 0.40 / height, left=0.075, right=0.805,
                        hspace=0.12)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ('png', 'pdf'):
        path = args.out_dir / f'membrane_grid.{suffix}'
        fig.savefig(path)
        written.append(path)
    plt.close(fig)

    for bits in BITS:
        points, scale = levels(bits)
        print(f'{bits} bit: {len(points)} levels, step {scale:.4f}, '
              f'{int(((points > 0) & (points < THRESHOLD)).sum())} below threshold, '
              f'first four {np.round(points[:4], 4)}')
    print('\n' + '\n'.join(str(p) for p in written))


if __name__ == '__main__':
    main()