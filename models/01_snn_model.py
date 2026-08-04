import torch
from torch import tensor
import torch.nn as nn
import snntorch as snn
from snntorch import spikeplot as splt

'''
Instantiate and construct main class: GWGlitchSNN
'''

class GWGlitchSNN(torch.nn.Module):
    '''
    Init method
    '''

    def __init__(self):
        super().__init__()

        beta = 0.85
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

        '''
        Spectrograms of 224 x 224 pixel.
        4 conv layer, stride = 2 meaning halfing the image dimension at each step
        therefore: 224 → 112 → 56 → 28 → 14.
        The last convolusion has output of 128 channels (feature maps)
        multiplying height, width and channels: 14 * 14 * 128 = 25088
        '''

        self.linear = nn.Linear(in_features = 25088, out_features= 4)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad)

    '''
    Forward method
    '''    
    
    def forward(self, x):
        # Define membrane potentials

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spike_out = []  # list to accumulated fired spiekes

        # Iterate through time
        for step in range(x.size(0)):

            cur1 = self.conv1(x[step])
            spike1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.conv2(spike1) 
            spike2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.conv3(spike2) 
            spike3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.conv4(spike3) 
            spike4, mem4 = self.lif4(cur4, mem4)

            flat = self.flatten(spike4)

            cur_out = self.linear(flat)

            spike_final, mem_out = self.lif_out(cur_out, mem_out)

            spike_out.append(spike_final)

        return torch.stack(spike_out, dim=0)

'''
Sanity check: should give 10 time step, for 8 images, and 4 class probabilities
'''
if __name__ == "__main__":
    net = GWGlitchSNN()
    dummy = torch.rand(10, 8, 3, 224, 224)

    print("Initialize forward pass")
    try:
        output = net(dummy)
        print(f"Chek completed - output shapes:{output.shape}")
    except Exception as e:
        print("Dimensional error")
        print(e)