import torch
import torch.nn as nn
from laplacewave import Laplace_fast as fast

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.p1_1 = nn.Sequential(#nn.Conv1d(1, 50, kernel_size=18, stride=2),
                                  fast(out_channels=50, kernel_size=18, stride=2),
                                  nn.BatchNorm1d(50),
                                  # nn.PReLU(50),
                                  nn.ReLU()

                                  )
        self.p1_2 = nn.Sequential(nn.Conv1d(50, 30, kernel_size=10, stride=2),
                                  nn.BatchNorm1d(30),
                                  #nn.PReLU(30)
                                  nn.ReLU()

                                  )
        self.p1_3 = nn.MaxPool1d(kernel_size=2)
        self.p2_1 = nn.Sequential(#nn.Conv1d(1, 50, kernel_size=6, stride=1),
                                  fast(out_channels=50, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(50),
                                  #nn.PReLU(50)
                                  nn.ReLU()

                                  )
        self.p2_2 = nn.Sequential(nn.Conv1d(50, 40, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(40),
                                  #nn.PReLU(40)
                                  nn.ReLU()

                                  )
        self.p2_3 = nn.MaxPool1d(kernel_size=2)
        self.p2_4 = nn.Sequential(nn.Conv1d(40, 30, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(30),
                                  #nn.PReLU(30)
                                  nn.ReLU()

                                  )
        self.p2_5 = nn.Sequential(nn.Conv1d(30, 30, kernel_size=8, stride=2),
                                  nn.BatchNorm1d(30),
                                  #nn.PReLU(30)
                                  nn.ReLU()

                                 )  # PRelu
        self.p2_6 = nn.MaxPool1d(kernel_size=2)
        self.p3_1 = nn.Sequential(nn.GRU(124, 64, bidirectional=True))
        self.p3_3 = nn.Sequential(nn.AdaptiveAvgPool1d(1))
        self.p4 = nn.Sequential(nn.Linear(30, 4))

    def forward(self, x):
        p1 = self.p1_3(self.p1_2(self.p1_1(x)))
        p2 = self.p2_6(self.p2_5(self.p2_4(self.p2_3(self.p2_2(self.p2_1(x))))))
        encode = torch.mul(p1, p2)    #b,c,d
        p3_0 = encode.permute(1, 0, 2)  #c,b,d
        p3_1, _ = self.p3_1(p3_0)
        p3_11 = p3_1.permute(1, 0, 2)#[:,-1,:] # 取得最后的一次输出
        p3_12 = self.p3_3(p3_11).squeeze() #无论信号长度多少，都只输出(批次×最后一个卷积块的输出通道)的张量
        p4 = self.p4(p3_12) #用全连接层分成10分类
        return p4
