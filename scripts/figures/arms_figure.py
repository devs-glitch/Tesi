'''
The three quantization arms, as which parts of the datapath are quantized
'''

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT_DIR = Path('results/Figures')

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d8d7d2'
W_EDGE, W_FILL = '#2a78d6', '#dce9f8'      # weight quantiser
U_EDGE, U_FILL = '#eb6834', '#fbe2d6'      # membrane quantiser
OFF_EDGE, OFF_FILL = '#b9b8b3', '#f4f3f0'  # not quantized in this arm

CHANNELS = (3, 16, 32, 64, 128)

# name, weights quantized, hidden membrane quantized
ARMS = (('Weight-only (W)', True, False),
        ('Membrane-only (U)', False, True),
        ('Joint (W + U)', True, True))

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': None,
    'figure.facecolor': 'white', 'font.size': 9,
    'text.color': INK, 'axes.labelcolor': INK,
})


def cell(ax, x, y, w, h, on, edge_on, fill_on, label, sub=None):
    edge, fill = (edge_on, fill_on) if on else (OFF_EDGE, OFF_FILL)
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0,rounding_size=0.5',
                                facecolor=fill, edgecolor=edge, linewidth=1.1,
                                zorder=3))
    colour = INK if on else MUTED
    ax.text(x + w / 2, y + h / 2 + (0.85 if sub else 0), label, ha='center',
            va='center', fontsize=7.2, color=colour, zorder=4)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 1.15, sub, ha='center', va='center',
                fontsize=6.0, color=MUTED, zorder=4)


def draw_arm(ax, title, weights_on, membrane_on, first):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 20)
    ax.axis('off')

    col_w, cell_h = 13.2, 5.6
    y_top, y_bot = 11.6, 5.2
    xs = [14.0 + i * 15.8 for i in range(5)]

    # Input.
    ax.add_patch(Rectangle((3.0, 7.3), 6.4, 6.4, facecolor='#f4f3f0',
                           edgecolor=MUTED, linewidth=0.9, zorder=3))
    ax.text(6.2, 5.6, 'input', ha='center', va='top', fontsize=6.4, color=MUTED)

    for i, x in enumerate(xs):
        last = i == 4
        if last:
            top, top_sub = 'Linear', None
            bottom, bottom_sub = 'LIF', 'output'
            # lif_out is excluded from QUANTISED_LIF in every arm
            membrane_here = False
        else:
            top = f'Conv {CHANNELS[i]}$\\rightarrow${CHANNELS[i + 1]}'
            top_sub = None
            bottom, bottom_sub = 'LIF', None
            membrane_here = membrane_on

        cell(ax, x, y_top, col_w, cell_h, weights_on, W_EDGE, W_FILL, top, top_sub)
        cell(ax, x, y_bot, col_w, cell_h, membrane_here, U_EDGE, U_FILL,
             bottom, bottom_sub)

        src = 9.8 if i == 0 else xs[i - 1] + col_w
        ax.add_patch(FancyArrowPatch((src, y_bot + cell_h / 2),
                                     (x - 0.5, y_bot + cell_h / 2),
                                     arrowstyle='-|>', mutation_scale=7,
                                     color=MUTED, linewidth=0.9, zorder=2,
                                     shrinkA=0, shrinkB=0))
        if first:
            ax.text(x + col_w / 2, 19.4,
                    'Readout' if last else f'Block {i + 1}', ha='center',
                    va='top', fontsize=7.0, color=MUTED)

    ax.text(0.0, 2.3, title, ha='left', va='bottom', fontsize=8.6,
            weight='bold')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    args = parser.parse_args()

    fig, axes = plt.subplots(len(ARMS), 1, figsize=(7.2, 4.3))
    height = fig.get_figheight()
    for i, (ax, (title, w_on, u_on)) in enumerate(zip(axes, ARMS)):
        draw_arm(ax, title, w_on, u_on, first=(i == 0))

    fig.suptitle('What each arm quantizes', x=0.004, y=1 - 0.05 / height,
                 ha='left', va='top', fontsize=10.5, weight='bold')
    legend = ((W_FILL, W_EDGE, 'weight quantiser'),
              (U_FILL, U_EDGE, 'membrane quantiser'),
              (OFF_FILL, OFF_EDGE, 'full precision'))
    for i, (fill, edge, label) in enumerate(legend):
        x = 0.345 + i * 0.222
        fig.patches.append(plt.Rectangle((x, 0.947), 0.015, 0.022,
                                         transform=fig.transFigure,
                                         facecolor=fill, edgecolor=edge,
                                         linewidth=1.1))
        fig.text(x + 0.021, 0.958, label, va='center', fontsize=7.2, color=INK)

    fig.tight_layout()
    fig.subplots_adjust(top=1 - 0.34 / height, hspace=0.06)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ('png', 'pdf'):
        path = args.out_dir / f'quantization_arms.{suffix}'
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    print('\n'.join(str(p) for p in written))


if __name__ == '__main__':
    main()