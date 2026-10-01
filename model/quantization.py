'''
Quantisation-aware training by fake quantisation for the three arms of the grid
'''

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

# Fixed by the clipping study
MEMBRANE_RANGE = (0.0, 2.0)

# hidden LIF layers
QUANTISED_LIF = ('lif1', 'lif2', 'lif3', 'lif4')

EPS = 1e-12

def signed_levels(bits):
    # Largest level of the symmetric signed grid: qmax = 2^(b-1) - 1
    return 2 ** (bits - 1) - 1


def grid(value_range, bits):
    '''
    Scale and integer bounds for a range, at a bit-width

    scheme is chosen by the sign of the range:

      lo >= 0   unsigned, levels 0 .. 2^b - 1,  step = hi / (2^b - 1)
      lo <  0   symmetric signed, levels -(2^(b-1)-1) .. +(2^(b-1)-1),
                step = max(|lo|, hi) / (2^(b-1) - 1)

    '''    
    lo, hi = value_range
    if lo >= 0:
        qmax = 2 ** bits - 1
        return hi / qmax, 0, qmax
    qmax = signed_levels(bits)
    return max(abs(lo), abs(hi)) / qmax, -qmax, qmax


def ste_round(x):
    # Round in the forward pass, identity in the backward pass
    return (x.round() - x).detach() + x


def fake_quantise(x, scale, qmin, qmax):
    '''
    Round to the grid, clamped to the integer bounds
    Rounding happens before the clamp
    '''
    return torch.clamp(ste_round(x / scale), qmin, qmax) * scale


# Weights

class PerChannelWeightQuantiser(nn.Module):
    # Parametrisation that fake-quantises a weight tensor, per output channel

    def __init__(self, bits):
        super().__init__()
        self.bits = bits
        self.qmax = signed_levels(bits)

    def forward(self, weight):
        # axis 0 is the output channel for both Conv2d [out, in, kh, kw]
        # and Linear [out, in]
        max_abs = weight.detach().abs().flatten(1).max(dim=1).values
        scale = (max_abs / self.qmax).clamp(min=EPS)
        scale = scale.view([-1] + [1] * (weight.dim() - 1))
        return fake_quantise(weight, scale, -self.qmax, self.qmax)


# Membrane
class MembraneQuantiser:
    # forward hooks that quantise the membrane potential of the hidden LIFs

    def __init__(self, net, bits, membrane_range=MEMBRANE_RANGE,
                 names=QUANTISED_LIF):
        self.bits = bits
        self.range = tuple(membrane_range)
        self.scale, self.qmin, self.qmax = grid(self.range, bits)
        self.names = tuple(names)
        self.handles = []

        for name in self.names:
            module = getattr(net, name)
            self.handles.append(module.register_forward_hook(self._hook))

    def _hook(self, module, inputs, output):
        spk, mem = output
        return spk, fake_quantise(mem, self.scale, self.qmin, self.qmax)

    @property
    def n_levels(self):
        return self.qmax - self.qmin + 1

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

# Application
class QuantisedModel:

    def __init__(self, net, weight_bits=None, membrane_bits=None,
                 membrane_range=MEMBRANE_RANGE, exclude=()):
        self.net = net
        self.weight_bits = weight_bits
        self.membrane_bits = membrane_bits
        self.membrane_range = tuple(membrane_range)
        self.exclude = tuple(exclude)
        self._quantised_modules = []
        self._membrane = None

        if weight_bits is not None:
            for name, module in net.named_modules():
                if not isinstance(module, (nn.Conv2d, nn.Linear)):
                    continue
                if name in self.exclude:
                    continue
                parametrize.register_parametrization(
                    module, 'weight', PerChannelWeightQuantiser(weight_bits))
                self._quantised_modules.append((name, module))

        if membrane_bits is not None:
            self._membrane = MembraneQuantiser(net, membrane_bits,
                                               self.membrane_range)

    def bake(self):
        """Fold the quantised weights into ordinary parameters, then detach.

        Leaves the network holding plain 'weight' tensors whose values sit on the
        quantisation grid
        """
        for _, module in self._quantised_modules:
            if parametrize.is_parametrized(module, 'weight'):
                parametrize.remove_parametrizations(
                    module, 'weight', leave_parametrized=True)
        self._quantised_modules = []

    def remove(self):
        # restore the network to full precision weights as they were

        for _, module in self._quantised_modules:
            if parametrize.is_parametrized(module, 'weight'):
                parametrize.remove_parametrizations(
                    module, 'weight', leave_parametrized=False)
        self._quantised_modules = []

        if self._membrane is not None:
            self._membrane.remove()
            self._membrane = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    # Reporting
    @property
    def arm(self):
    
        if self.weight_bits and self.membrane_bits:
            return 'joint'
        if self.weight_bits:
            return 'weight-only'
        if self.membrane_bits:
            return 'membrane-only'
        return 'fp32'

    def config(self):
    
        record = {
            'arm': self.arm,
            'weight_bits': self.weight_bits,
            'membrane_bits': self.membrane_bits,
            'weight_granularity': 'per-channel' if self.weight_bits else None,
            'quantised_weight_layers': [n for n, _ in self._quantised_modules],
            'quantised_lif_layers': list(self._membrane.names) if self._membrane else [],
        }
        if self.membrane_bits:
            record.update({
                'membrane_range': list(self.membrane_range),
                'membrane_scheme': ('unsigned' if self.membrane_range[0] >= 0
                                    else 'symmetric signed'),
                'membrane_step': self._membrane.scale,
                'membrane_levels': self._membrane.n_levels,
            })
        return record

    def describe(self):
        # One line naming the arm and the grids it implies
        parts = [f'arm={self.arm}']
        if self.weight_bits:
            parts.append(f'W={self.weight_bits}b, per-channel, '
                         f'{2 * signed_levels(self.weight_bits) + 1} levels, '
                         f'{len(self._quantised_modules)} layers')
        if self.membrane_bits:
            m = self._membrane
            scheme = 'unsigned' if self.membrane_range[0] >= 0 else 'signed'
            parts.append(f'U={self.membrane_bits}b, {scheme} '
                         f'[{self.membrane_range[0]:g}, {self.membrane_range[1]:g}], '
                         f'{m.n_levels} levels, step {m.scale:.4f} '
                         f'({100 * m.scale:.1f}% of threshold)')
        return ' | '.join(parts)

    def weight_steps(self):
        # Mean quantisation step per layer
        steps = {}
        for name, module in self._quantised_modules:
            original = module.parametrizations.weight.original
            max_abs = original.detach().abs().flatten(1).max(dim=1).values
            steps[name] = float((max_abs / signed_levels(self.weight_bits)).mean())
        return steps


def apply_quantisation(net, weight_bits=None, membrane_bits=None,
                       membrane_range=MEMBRANE_RANGE, exclude=()):
    # attach fake quantisation to a network and return the handle

    return QuantisedModel(net, weight_bits, membrane_bits, membrane_range,
                          exclude)


# Self-checks

def check_grid(verbose=True):
    '''
      zero is exactly representable        otherwise every resting neuron is biased
      the error never exceeds half a step  otherwise the grid is not what it claims
      the level count matches the scheme   otherwise the bit-width is not what it says
      the straight-through gradient is 1   otherwise training cannot recover
      the range ends are representable     otherwise the clipping is not what was tested
    '''

    ok = True
    for value_range in ((0.0, 2.0), (-2.0, 2.0)):
        lo, hi = value_range
        scheme = 'unsigned' if lo >= 0 else 'signed  '
        for bits in (8, 4, 2):
            scale, qmin, qmax = grid(value_range, bits)
            expected_levels = qmax - qmin + 1

            x = torch.linspace(lo, hi, 10001)
            q = fake_quantise(x, scale, qmin, qmax)

            zero_exact = float(fake_quantise(torch.zeros(1), scale, qmin, qmax)) == 0.0
            max_error = float((q - x).abs().max())
            n_levels = len(torch.unique(q))
            ends_exact = (abs(float(q[0]) - lo) < 1e-5
                          and abs(float(q[-1]) - hi) < 1e-5)

            g = torch.linspace(lo, hi, 101, requires_grad=True)
            fake_quantise(g, scale, qmin, qmax).sum().backward()
            gradient_ok = bool(torch.allclose(g.grad, torch.ones_like(g.grad)))

            passed = (zero_exact and max_error <= scale / 2 + 1e-6
                      and n_levels == expected_levels and gradient_ok
                      and ends_exact)
            ok = ok and passed

            if verbose:
                print(f'  {scheme} [{lo:g}, {hi:g}] {bits}b: '
                      f'step {scale:.4f} ({100 * scale:6.2f}% of threshold)'
                      f' | {n_levels:3d} levels (expected {expected_levels:3d})'
                      f' | err {max_error:.5f} <= {scale / 2:.5f}'
                      f' | zero {zero_exact} | ends {ends_exact}'
                      f' | STE {gradient_ok}'
                      f' | {"OK" if passed else "FAILED"}')
    return ok


def check_threshold_resolvable(verbose=True):
    '''
    A membrane grid is only useful if it can represent a potential
    in a way that at least one level falls strictly between 0 and the threshold
    '''

    threshold = 1.0
    ok = True
    for value_range in ((0.0, 2.0), (-2.0, 2.0)):
        for bits in (8, 4, 2):
            scale, qmin, qmax = grid(value_range, bits)
            points = torch.arange(qmin, qmax + 1, dtype=torch.float32) * scale
            between = int(((points > 0) & (points < threshold)).sum())
            scheme = 'unsigned' if value_range[0] >= 0 else 'signed  '
            if verbose:
                print(f'  {scheme} [{value_range[0]:g}, {value_range[1]:g}] '
                      f'{bits}b: {between} level(s) strictly between rest and '
                      f'threshold'
                      + ('' if between else '   <- threshold NOT resolvable'))
            if value_range[0] >= 0:
                ok = ok and between >= 1
    return ok


def check_idempotence(verbose=True):
    torch.manual_seed(0)
    ok = True
    for bits in (8, 4, 2):
        w = torch.randn(16, 8, 3, 3)
        quantiser = PerChannelWeightQuantiser(bits)
        once = quantiser(w)
        twice = quantiser(once)
        drift = float((twice - once).abs().max())
        passed = drift < 1e-6
        ok = ok and passed
        if verbose:
            print(f'  {bits} bit: max drift on re-quantisation {drift:.3e}'
                  f' | {"OK" if passed else "FAILED"}')
    return ok


if __name__ == '__main__':
    print('grids:')
    grid_ok = check_grid()

    print('\nthreshold resolvability:')
    threshold_ok = check_threshold_resolvable()

    print('\nidempotence of the weight quantiser:')
    bake_ok = check_idempotence()

    if not (grid_ok and threshold_ok and bake_ok):
        raise SystemExit('\nself-check failed')
    print('\nall properties hold')