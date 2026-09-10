import torch
from torch import tensor
import torch.nn as nn
import snntorch as snn
from snntorch import spikeplot as splt

'''
Instantiate and construct main class: GWGlitchSNN
'''


def conv_output_size(size, kernel_size=5, stride=2, padding=2):
    return (size + 2 * padding - kernel_size) // stride + 1


def flattened_features(input_size, n_conv_layers=4, out_channels=128):
    side = input_size
    for _ in range(n_conv_layers):
        side = conv_output_size(side)
    # la formula satura a 1 e non arriva mai a 0, quindi il controllo utile e'
    # sulla degenerazione: una feature map finale 1x1 ha perso ogni struttura
    # spaziale, e le mappe SAM diventerebbero un singolo pixel.
    if side < 2:
        raise ValueError(
            f'input_size={input_size} too small {n_conv_layers} ')
    return side * side * out_channels, side


class GWGlitchSNN(torch.nn.Module):
    '''
    Init method
    '''

    def __init__(self, beta=0.5, input_size=224):
        super().__init__()

        self.input_size = input_size

        # Address non-differentiability by defining a surrogate gradient with arctan for the BBP
        spike_grad = snn.surrogate.atan()

        # Layer configuration
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 5, stride = 2, padding = 2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 2, padding = 2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 2, padding = 2)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 2, padding = 2)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.flatten = nn.Flatten()

        in_features, self.final_side = flattened_features(input_size)

        self.linear = nn.Linear(in_features = in_features, out_features= 4)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad)

    '''
    Forward method
    '''    
    
    def forward(self, x):
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            raise ValueError(
                f'input size of dataloader not same of input size of model')

        # Define membrane potentials

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spike_out = []  # list to accumulated fired spiekes
        spike1_out = []
        spike2_out = []
        spike3_out = []
        spike4_out = []

        # Iterate through time
        for step in range(x.size(0)):

            cur1 = self.conv1(x[step])
            spike1, mem1 = self.lif1(cur1, mem1)
            spike1_out.append(spike1)

            cur2 = self.conv2(spike1) 
            spike2, mem2 = self.lif2(cur2, mem2)
            spike2_out.append(spike2)

            cur3 = self.conv3(spike2) 
            spike3, mem3 = self.lif3(cur3, mem3)
            spike3_out.append(spike3)

            cur4 = self.conv4(spike3) 
            spike4, mem4 = self.lif4(cur4, mem4)
            spike4_out.append(spike4)

            flat = self.flatten(spike4)

            cur_out = self.linear(flat)

            spike_final, mem_out = self.lif_out(cur_out, mem_out)

            spike_out.append(spike_final)

        spike_out_stacked = torch.stack(spike_out, dim=0)
        spike1_stacked = torch.stack(spike1_out, dim=0)
        spike2_stacked = torch.stack(spike2_out, dim=0)
        spike3_stacked = torch.stack(spike3_out, dim=0)
        spike4_stacked = torch.stack(spike4_out, dim=0)

        return spike_out_stacked, spike1_stacked, spike2_stacked, spike3_stacked, spike4_stacked


'''
Sanity check: forward pass for each resolution
'''
if __name__ == "__main__":
    for size in (224, 112, 64):
        in_features, side = flattened_features(size)
        print(f"\n{size} px -> feature map final {side}x{side} "
              f"-> in_features = {in_features}")

        net = GWGlitchSNN(input_size=size)
        dummy = torch.rand(10, 8, 3, size, size)

        try:
            output_tot, output1, output2, output3, output4 = net(dummy)
            print(f"  forward ok - output shape: {tuple(output_tot.shape)}")
            print(f"  parameters: {sum(p.numel() for p in net.parameters()):,}")
        except Exception as e:
            print(f"  Dimensional error: {e}")
