import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from math import pi
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
# setup_seed(1)

def Laplace(p):
    A = 0.08
    ep = 0.03
    tal = 0.1
    f = 80
    w = 2 * pi * f
    q = torch.tensor(1 - pow(ep, 2))
    # y = A * torch.exp((-ep / (torch.sqrt(q))) * (w * (p - tal))) * (-torch.sin(w * (p - tal)))
    y = A * torch.exp((-ep / (torch.sqrt(q))) * (w * (p - tal))) * (torch.sin(w * (p - tal)))  ##rgiht
    #y = A * torch.exp((-ep / (torch.sqrt(q))) * (w * (p - tal))) * torch.exp(((0-1j) * w * (p - tal)))
    return y

class Laplace_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1, stride=2, eps=0.1):#

        super(Laplace_fast, self).__init__()

        if in_channels != 1:

            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        # self.kernel_size = kernel_size - 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps

        # if kernel_size % 2 == 0:
        #     self.kernel_size = self.kernel_size + 1

        self.a_ = nn.Parameter(torch.linspace(0, 100, out_channels).view(-1, 1))
        # self.a_ = nn.Parameter(torch.linspace(0, 100, out_channels)).view(-1, 1)
        self.b_ = nn.Parameter(torch.linspace(0, 100, out_channels).view(-1, 1))
        # self.a_ = nn.Parameter(torch.randn(out_channels).view(-1, 1))
        # # self.a_ = nn.Parameter(torch.linspace(0, 100, out_channels)).view(-1, 1)
        # self.b_ = nn.Parameter(torch.randn(out_channels).view(-1, 1))
        self.bias = nn.Parameter(torch.zeros(1, self.out_channels).squeeze().cuda())
        # self.weight_norm_factor = 1.0
        # self.transposed = False
    # def normalized_weight(self, weight):
    #     weight_norm = weight.pow(2)
    #     for i in range(self.kernel_size):
    #         weight_norm = weight_norm.sum(i + 2, keepdim=True)
    #     if self.transposed:
    #         weight_norm = weight_norm.sum(0, keepdim=True) * self.weight_norm_factor
    #     else:
    #         weight_norm = weight_norm.sum(1, keepdim=True)
    #     weight_norm = weight_norm.sqrt().add(1e-8)
    #     weight = weight.div(weight_norm)
    #     if self.scale is not None:
    #         scale_unsqueeze = self.scale
    #         if self.transposed:
    #             scale_unsqueeze = scale_unsqueeze.unsqueeze(0)
    #         else:
    #             scale_unsqueeze = scale_unsqueeze.unsqueeze(1)
    #         for i in range(2, weight.dim()):
    #             scale_unsqueeze = scale_unsqueeze.unsqueeze(i)
    #         weight = weight.mul(scale_unsqueeze)
    #     return weight


    def forward(self, waveforms):

        time_disc = torch.linspace(0, 1, steps=int((self.kernel_size)))
        # print((time_disc.cuda() - self.b_.cuda()).size())
        # print((self.a_.cuda() + self.eps).size())
        p1 = (time_disc.cuda() - self.b_.cuda()) / (self.a_.cuda() + self.eps)#
        #p1 = (time_disc.cuda() - self.b_.cuda()) / self.a_.cuda()
        laplace_filter = Laplace(p1)
        self.filters = laplace_filter.unsqueeze(1)
        # self.filters = (laplace_filter).view(self.out_channels, 1, self.kernel_size).cuda()  #(70,1,85)
        # filters_norm = torch.norm(self.filters, p=2, dim=0)  #(1,85)
        # self.filters = self.filters * torch.norm_except_dim(self.filters, 2, dim=0) / (torch.norm(self.filters, p=2, dim=0))  #(70,1,85)
        # m = F.conv1d(waveforms, self.filters, bias=self.bias, stride=self.stride, padding=1)
        # n = F.conv1d(waveforms, self.filters, stride=self.stride, padding=1)
        return F.conv1d(waveforms, self.filters,  bias=self.bias, stride=self.stride, padding=0)#
        #return F.conv1d(waveforms, self.filters, stride=self.stride, padding=1)

if __name__ == '__main__':
    input = torch.randn(2, 1, 1024).cuda()
    model = Laplace_fast(out_channels=70, kernel_size=85, stride=1).cuda()
    print(model)
    for param in model.parameters():
        print(type(param.data), param.size())
    output = model(input)
    print(output.size())

def Morlet(p, c):

    y = c * torch.exp(-torch.pow(p, 2) / 2) * torch.cos(5 * p)
    return y

class Morlet_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1, stride=2):

        super(Morlet_fast, self).__init__()

        if in_channels != 1:

            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1
        self.stride = stride

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.a_ = nn.Parameter(torch.linspace(1, 100, out_channels)).view(-1, 1)

        self.b_ = nn.Parameter(torch.linspace(0, 100, out_channels)).view(-1, 1)

    def forward(self, waveforms):

        time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1,
                                         steps=int((self.kernel_size / 2)))

        time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1,
                                        steps=int((self.kernel_size / 2)))

        p1 = (time_disc_right.cuda() - self.b_.cuda()) / self.a_.cuda()
        p2 = (time_disc_left.cuda() - self.b_.cuda()) / self.a_.cuda()
        C = pow(pi, 0.25)
        D = C / torch.sqrt(self.a_.cuda())     ##一个值得探讨的点
        #D = C / self.a_.cuda()
        Morlet_right = Morlet(p1, D)
        Morlet_left = Morlet(p2, D)
        Morlet_filter = torch.cat([Morlet_left, Morlet_right], dim=1)  # 40x1x250
        self.filters = (Morlet_filter).view(self.out_channels, 1, self.kernel_size).cuda()
        return F.conv1d(waveforms, self.filters, stride=self.stride, padding=1, dilation=1, bias=None, groups=1)


def Morlet1(p,C):
    # C = pow(pi, 0.25)
    y = C * torch.exp(-torch.pow(p, 2) / 2) * torch.cos(2 * pi * p)
    return y

class Morlet_fast1(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1, stride=2):

        super(Morlet_fast1, self).__init__()

        if in_channels != 1:

            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1
        self.stride = stride

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.a_ = nn.Parameter(torch.linspace(1, 100, out_channels)).view(-1, 1)

        self.b_ = nn.Parameter(torch.linspace(0, 100, out_channels)).view(-1, 1)

    def forward(self, waveforms):

        time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1,
                                         steps=int((self.kernel_size / 2)))

        time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1,
                                        steps=int((self.kernel_size / 2)))

        p1 = time_disc_right.cuda() - self.b_.cuda() / self.a_.cuda()
        p2 = time_disc_left.cuda() - self.b_.cuda() / self.a_.cuda()

        Morlet_right = Morlet1(p1)
        Morlet_left = Morlet1(p2)

        Morlet_filter = torch.cat([Morlet_left, Morlet_right], dim=1)  # 40x1x250

        self.filters = (Morlet_filter).view(self.out_channels, 1, self.kernel_size).cuda()

        return F.conv1d(waveforms, self.filters, stride=self.stride, padding=1, dilation=1, bias=None, groups=1)

def Gaussian(p):
    # y = D * torch.exp(-torch.pow(p, 2))
    F0 = (2./pi)**(1./4.) * torch.exp(-torch.pow(p, 2))
    y = -2 / (3 ** (1 / 2)) * (-1 + 2 * p ** 2) * F0
    return y

class Gaussian_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1, stride=2):

        super(Gaussian_fast, self).__init__()

        if in_channels != 1:

            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1
        self.stride = stride

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.a_ = nn.Parameter(torch.linspace(1, 100, out_channels)).view(-1, 1)

        self.b_ = nn.Parameter(torch.linspace(0, 100, out_channels)).view(-1, 1)

    def forward(self, waveforms):

        time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1,
                                         steps=int((self.kernel_size / 2)))

        time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1,
                                        steps=int((self.kernel_size / 2)))

        p1 = time_disc_right.cuda() - self.b_.cuda() / self.a_.cuda()
        p2 = time_disc_left.cuda() - self.b_.cuda() / self.a_.cuda()
        # D = 1 / torch.sqrt(self.a_.cuda())
        Gaussian_right = Gaussian(p1)
        Gaussian_left = Gaussian(p2)

        Gaussian_filter = torch.cat([Gaussian_left, Gaussian_right], dim=1)  # 40x1x250

        self.filters = (Gaussian_filter).view(self.out_channels, 1, self.kernel_size).cuda()

        return F.conv1d(waveforms, self.filters, stride=self.stride, padding=1, dilation=1, bias=None, groups=1)

def Mexh(p):
    # p = 0.04 * p  # 将时间转化为在[-5,5]这个区间内
    y = (2/torch.sqrt(torch.Tensor([3]).cuda())*(1/(torch.pow(torch.Tensor([pi]).cuda(), (1/4)))))*(1 - torch.pow(p, 2)) * torch.exp(-torch.pow(p, 2) / 2)

    return y

class Mexh_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1, stride=2):

        super(Mexh_fast, self).__init__()

        if in_channels != 1:

            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels

        self.kernel_size = kernel_size - 1
        self.stride = stride

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1


        self.a_ = nn.Parameter(torch.linspace(1, 100, out_channels)).view(-1, 1)

        self.b_ = nn.Parameter(torch.linspace(0, 100, out_channels)).view(-1, 1)

    def forward(self, waveforms):

        time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1,
                                         steps=int((self.kernel_size / 2)))

        time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1,
                                        steps=int((self.kernel_size / 2)))

        p1 = time_disc_right.cuda() - self.b_.cuda() / self.a_.cuda()

        p2 = time_disc_left.cuda() - self.b_.cuda() / self.a_.cuda()

        Mexh_right = Mexh(p1)
        Mexh_left = Mexh(p2)

        Mexh_filter = torch.cat([Mexh_left, Mexh_right], dim=1)  # 40x1x250

        self.filters = (Mexh_filter).view(self.out_channels, 1, self.kernel_size).cuda()

        return F.conv1d(waveforms, self.filters, stride=self.stride, padding=1, dilation=1, bias=None, groups=1)