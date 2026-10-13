'''
Shared drawing conventions for every figure of the thesis
'''

from pathlib import Path

import matplotlib.pyplot as plt

FIG_DIR = Path('results/Figures')

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d8d7d2'
BLUE, ORANGE, GREEN, PURPLE = '#2a78d6', '#eb6834', '#1baf7a', '#8b5fd6'
BLUE_FILL = '#dce9f8'


def use_thesis_style():
    plt.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': None,
        'figure.facecolor': 'white', 'font.size': 9,
        'text.color': INK, 'axes.labelcolor': INK,
        'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
    })


def frame(fig, title, note=None, pad=1.20):
    # Left-aligned header, with 'pad' inches reserved above the axes for it
    height = fig.get_figheight()
    fig.suptitle(title, x=0.006, y=1 - 0.06 / height, ha='left', va='top',
                 fontsize=10.5, weight='bold')
    if note:
        fig.text(0.006, 1 - 0.30 / height, note, ha='left', va='top',
                 fontsize=7.8, color=MUTED)
    fig.tight_layout()
    fig.subplots_adjust(top=1 - pad / height)


def clean_axes(ax, grid_axis='y'):
    #  Recessive grid behind the data, no box around it
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def save(fig, stem, out_dir=FIG_DIR):
    # PNG for reading and PDF for the thesis
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ('png', 'pdf'):
        path = out_dir / f'{stem}.{suffix}'
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    print('\n'.join(str(p) for p in written))