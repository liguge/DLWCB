import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from math import pi


def Laplace(p):
    A = 0.08
    ep = 0.03
    tal = 0.1
    f = 80
    w = 2 * pi * f
    q = torch.tensor(1 - pow(ep, 2))
    y = A * torch.exp((-ep / (torch.sqrt(q))) * (w * (p - tal))) * (torch.sin(w * (p - tal)))
    return y

class Laplace_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1, stride=2):

        super(Laplace_fast, self).__init__()

        if in_channels != 1:

            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.a_ = nn.Parameter(torch.linspace(1, 100, out_channels).view(-1, 1))
        self.b_ = nn.Parameter(torch.linspace(0, 100, out_channels).view(-1, 1))
        self.bias = nn.Parameter(torch.zeros(1, self.out_channels).squeeze().cuda())



    def forward(self, waveforms):

        time_disc = torch.linspace(0, 1, steps=int((self.kernel_size)))
        p1 = (time_disc.cuda() - self.b_.cuda()) / (self.a_.cuda())
        laplace_filter = Laplace(p1)
        self.filters = laplace_filter.unsqueeze(1)

        return F.conv1d(waveforms, self.filters,  bias=self.bias, stride=self.stride, padding=0)#

if __name__ == '__main__':
    input = torch.randn(2, 1, 1024).cuda()
    model = Laplace_fast(out_channels=70, kernel_size=85, stride=1).cuda()
    print(model)
    for param in model.parameters():
        print(type(param.data), param.size())
    output = model(input)
    print(output.size())
