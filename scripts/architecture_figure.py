'''
Schematic of the GWGlitchSNN datapath
'''

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT_DIR = Path('results/Figures')

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d8d7d2'
W_EDGE, W_FILL = '#2a78d6', '#dce9f8'      # weight-quantiser target
U_EDGE, U_FILL = '#eb6834', '#fbe2d6'      # membrane-quantiser target
TAP = '#8b5fd6'                            # the layer the SAM maps are read at

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': None,
    'figure.facecolor': 'white', 'font.size': 8,
    'text.color': INK, 'axes.labelcolor': INK,
})

CHANNELS = (3, 16, 32, 64, 128)
N_CLASSES = 4
SAM_LAYER = 2                              # config/baseline_manifest.json


def conv_output_size(size, kernel_size=5, stride=2, padding=2):
    '''Identical to model.snn_model.conv_output_size.'''
    return (size + 2 * padding - kernel_size) // stride + 1


def spatial_sides(input_size, n_blocks=4):
    sides, side = [], input_size
    for _ in range(n_blocks):
        side = conv_output_size(side)
        sides.append(side)
    return sides


def parameter_count(sides):
    # Convolutions plus the readout, biases included
    total = 0
    for cin, cout in zip(CHANNELS[:-1], CHANNELS[1:]):
        total += cout * cin * 5 * 5 + cout
    flat = CHANNELS[-1] * sides[-1] * sides[-1]
    return total + flat * N_CLASSES + N_CLASSES, flat



def box(ax, x, y, w, h, fill, edge, lw=1.2, radius=0.6, z=3):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
                           facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z)
    ax.add_patch(patch)
    return patch


def arrow(ax, x0, y0, x1, y1, colour=MUTED, lw=1.1, style='-|>', z=2,
          connection='arc3,rad=0'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=9, color=colour, linewidth=lw,
                                 connectionstyle=connection, zorder=z,
                                 shrinkA=0, shrinkB=0))


def load_sample(path):
    '''
    Read an image and centre-crop it to a square

    Pass a spectrogram that has already been through the pipeline\'s crop: this
    does not reproduce the 470x550 panel extraction, it only squares whatever
    it is given.
    '''
    img = mpimg.imread(path)
    height, width = img.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    return img[top:top + side, left:left + side]


def frame_icon(ax, cx, cy, size, sample=None, colour=MUTED):
    extent = (cx - size / 2, cx + size / 2, cy - size / 2, cy + size / 2)
    if sample is not None:
        ax.imshow(sample, extent=extent, aspect='auto', zorder=3,
                  interpolation='antialiased')
    ax.add_patch(Rectangle((extent[0], extent[2]), size, size,
                           facecolor='none' if sample is not None else '#edece8',
                           edgecolor=colour, linewidth=0.9, zorder=4))


def draw_datapath(ax, input_size, sides, flat, beta, sample=None):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis('off')

    frame_icon(ax, 6.2, 23.4, 9.0, sample)
    ax.text(6.2, 16.0, f'$3 \\times {input_size} \\times {input_size}$',
            ha='center', va='top', fontsize=7.4)
    ax.text(6.2, 13.1, 'direct encoding', ha='center', va='top', fontsize=6.8,
            color=MUTED, style='italic')

    col_w, cell_h = 14.2, 6.5
    y_conv, y_lif = 26.0, 18.5
    xs = [14.0 + i * 17.0 for i in range(5)]

    for i, x in enumerate(xs):
        last = i == 4
        if last:
            top_a = 'Linear'
            top_b = f'${flat:,} \\rightarrow {N_CLASSES}$'.replace(',', '\\,')
            bottom_b = f'{N_CLASSES} neurons'
            shape = f'$[{N_CLASSES}]$'
            heading = 'Readout'
        else:
            cin, cout, side = CHANNELS[i], CHANNELS[i + 1], sides[i]
            top_a = f'Conv  ${cin} \\rightarrow {cout}$'
            top_b = r'$5 \times 5$,  s2,  p2'
            bottom_b = f'$\\beta = {beta}$'
            shape = f'$[{cout}, {side}, {side}]$'
            heading = f'Block {i + 1}'

        box(ax, x, y_conv, col_w, cell_h, W_FILL, W_EDGE)
        ax.text(x + col_w / 2, y_conv + cell_h / 2 + 1.1, top_a, ha='center',
                va='center', fontsize=7.5, zorder=4)
        ax.text(x + col_w / 2, y_conv + cell_h / 2 - 1.6, top_b, ha='center',
                va='center', fontsize=6.5, color=MUTED, zorder=4)

        box(ax, x, y_lif, col_w, cell_h, U_FILL, U_EDGE)
        ax.text(x + col_w / 2, y_lif + cell_h / 2 + 1.1, 'LIF', ha='center',
                va='center', fontsize=7.5, zorder=4)
        ax.text(x + col_w / 2, y_lif + cell_h / 2 - 1.6, bottom_b, ha='center',
                va='center', fontsize=6.5, color=MUTED, zorder=4)

        # Current from the convolution into the LIF layer that integrates it.
        arrow(ax, x + col_w / 2, y_conv, x + col_w / 2, y_lif + cell_h,
              colour=MUTED, lw=0.9)

        ax.text(x + col_w / 2, 37.4, heading, ha='center', va='top',
                fontsize=7.4, weight='bold')
        ax.text(x + col_w / 2, 16.0, shape, ha='center', va='top', fontsize=7.2)

        src = 11.2 if i == 0 else xs[i - 1] + col_w
        arrow(ax, src, y_lif + cell_h / 2, x - 0.5, y_lif + cell_h / 2)

        if i == SAM_LAYER - 1:
            arrow(ax, x + col_w / 2, 10.2, x + col_w / 2, 12.6, colour=TAP, lw=1.0)
            ax.text(x + col_w / 2, 9.4, 'SAM read here', ha='center', va='top',
                    fontsize=6.9, color=TAP, weight='bold')

    ax.text(0.0, 1.6, 'arctan surrogate gradient throughout; no batch '
            'normalization, pooling or dropout', ha='left', va='bottom',
            fontsize=6.8, color=MUTED)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)


def draw_unrolling(ax, time_steps, beta, sample=None):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis('off')

    xs = [15.0 + i * 7.6 for i in range(time_steps)]
    y_frame, y_net, y_spike = 22.0, 13.8, 6.4

    ax.text(0.0, 28.4, f'Unrolled over $T = {time_steps}$ steps', ha='left',
            va='top', fontsize=7.4, weight='bold')
    ax.text(22.0, 28.4, 'the same frame enters at every step, so the time axis '
            'carries no input-driven structure', ha='left', va='top',
            fontsize=6.8, color=MUTED, style='italic')

    for i, x in enumerate(xs):
        frame_icon(ax, x, y_frame, 4.4, sample)
        arrow(ax, x, y_frame - 2.4, x, y_net + 2.1, colour=MUTED, lw=0.8)
        box(ax, x - 2.9, y_net - 2.1, 5.8, 4.2, '#f7f6f3', MUTED, lw=1.0, radius=0.45)
        ax.text(x, y_net, f'$t_{{{i + 1}}}$', ha='center', va='center', fontsize=7.0)
        arrow(ax, x, y_net - 2.1, x, y_spike + 1.5, colour=MUTED, lw=0.8)
        ax.plot([x], [y_spike], marker='|', markersize=7, color=U_EDGE,
                markeredgewidth=1.6)

        if i < time_steps - 1:
            # The membrane is the only thing that crosses a time step
            arrow(ax, x + 2.9, y_net, xs[i + 1] - 2.9, y_net, colour=U_EDGE, lw=1.1)

    ax.text(xs[0] - 4.4, y_net, 'membrane', ha='right', va='center',
            fontsize=6.8, color=U_EDGE)
    ax.text(xs[0] - 4.4, y_spike, 'output spikes', ha='right', va='center',
            fontsize=6.8, color=MUTED)

    # Rate-coded readout: the counts carry the decision
    x_sum = xs[-1] + 7.0
    arrow(ax, xs[-1] + 1.4, y_spike, x_sum - 1.2, y_spike, colour=MUTED, lw=1.1)
    ax.text(x_sum, y_spike, r'$\sum_t S_{\mathrm{out}}[t]$', ha='left',
            va='center', fontsize=8.5)
    ax.text(x_sum, y_spike - 4.2, 'cross-entropy on the counts', ha='left',
            va='center', fontsize=6.8, color=MUTED)

    ax.text(0.0, 1.2, r'$U[t] = \beta\,U[t-1] + I[t] - \theta\,S[t-1]$,'
            f'   $\\beta = {beta}$,   threshold $\\theta = 1$', ha='left',
            va='bottom', fontsize=7.2, color=U_EDGE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-size', type=int, default=112)
    parser.add_argument('--time-steps', type=int, default=8)
    parser.add_argument('--beta', type=float, default=0.75)
    parser.add_argument('--sample', type=Path, default=None,
                        help='a preprocessed spectrogram to draw as the input; '
                             'without it the input square is left empty')
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    args = parser.parse_args()

    sample = load_sample(args.sample) if args.sample else None
    sides = spatial_sides(args.input_size)
    n_params, flat = parameter_count(sides)

    fig = plt.figure(figsize=(7.4, 5.1))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.42, 1.0], hspace=0.0,
                          left=0.006, right=0.994, top=0.905, bottom=0.02)
    ax_top, ax_bottom = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    draw_datapath(ax_top, args.input_size, sides, flat, args.beta, sample)
    draw_unrolling(ax_bottom, args.time_steps, args.beta, sample)

    fig.suptitle(f'GWGlitchSNN at {args.input_size} px', x=0.006,
                 y=0.985, ha='left', va='top', fontsize=11, weight='bold')

    legend = [
        (W_FILL, W_EDGE, 'weight quantiser'),
        (U_FILL, U_EDGE, 'membrane quantiser'),
    ]
    for i, (fill, edge, label) in enumerate(legend):
        x = 0.44 + i * 0.19
        fig.patches.append(plt.Rectangle((x, 0.944), 0.016, 0.019, transform=fig.transFigure,
                                         facecolor=fill, edgecolor=edge, linewidth=1.1))
        fig.text(x + 0.022, 0.9535, label, va='center', fontsize=7.2, color=INK)
    fig.text(0.996, 0.9535, f'{n_params:,} parameters'.replace(',', ' '),
             ha='right', va='center', fontsize=7.2, color=MUTED)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ('png', 'pdf'):
        path = args.out_dir / f'network_architecture_{args.input_size}px.{suffix}'
        fig.savefig(path)
        written.append(path)
    plt.close(fig)

    print(f'input {args.input_size} px -> sides {sides}, flat {flat:,}, '
          f'{n_params:,} parameters')
    print('\n'.join(str(p) for p in written))


if __name__ == '__main__':
    main()