import torch
import torch.nn as nn
import numpy as np
# import torch.optim as optim
# #from tensorboard import program
# from torch.optim.lr_scheduler import StepLR
from torch.optim import lr_scheduler
from wmodels1 import Laplace_fast as fast
from gdatasave import train_loader, test_loader
from early_stopping import EarlyStopping
from label_smoothing import LSR
from torch_optimizer import AdamP
# from loss_428 import GHMCC
import time
from tanhsoftplus import TanhSoft
# from torchsummary import summary
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)


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
        # self.p3_1 = nn.Sequential(nn.GRU(124, 150, bidirectional=True))
        self.p3_1 = nn.Sequential(nn.GRU(124, 64, bidirectional=True))
        # print(self.p3_1._all_weights)
        # self.p3_1.weight_ih_l0 = nn.Parameter(nn.init.orthogonal(self.p3_1.weight_ih_l0))
        # self.p3_1.weight_ih_l0_reverse = nn.Parameter(nn.init.orthogonal(self.p3_1.weight_ih_l0_reverse))
        # self.p3_1.weight_hh_l0 = nn.Parameter(nn.init.orthogonal(self.p3_1.weight_hh_l0))
        # self.p3_1.weight_hh_l0_reverse = nn.Parameter(nn.init.orthogonal(self.p3_1.weight_hh_l0_reverse))
        # self.p3_1.bias_hh_l0 = nn.Parameter(torch.zeros_like(self.p3_1.bias_hh_l0))
        # self.p3_1.bias_hh_l0_reverse = nn.Parameter(torch.zeros_like(self.p3_1.bias_hh_l0_reverse))
        # self.p3_1.bias_ih_l0 = nn.Parameter(torch.zeros_like(self.p3_1.bias_ih_l0))
        # self.p3_1.bias_ih_l0_reverse = nn.Parameter(torch.zeros_like(self.p3_1.bias_ih_l0_reverse))
        # print(self.p3_1.bias_hh_l0)
        # self.p3_2 = nn.Sequential(nn.Dropout(0.2),
        #             nn.GRU(300, 30, bidirectional=True))  #
        self.p3_3 = nn.Sequential(nn.AdaptiveAvgPool1d(1))
        self.p4 = nn.Sequential(nn.Linear(30, 4))

    def forward(self, x):
        p1 = self.p1_3(self.p1_2(self.p1_1(x)))
        p2 = self.p2_6(self.p2_5(self.p2_4(self.p2_3(self.p2_2(self.p2_1(x))))))
        encode = torch.mul(p1, p2)    #b,c,d
        p3_0 = encode.permute(1, 0, 2)  #c,b,d
        p3_1, _ = self.p3_1(p3_0)
        # p3_2, _ = self.p3_2(p3_1)
        p3_11 = p3_1.permute(1, 0, 2)#[:,-1,:] # 取得最后的一次输出
        p3_12 = self.p3_3(p3_11).squeeze() #无论信号长度多少，都只输出(批次×最后一个卷积块的输出通道)的张量
        p4 = self.p4(p3_12) #用全连接层分成10分类
        return p4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
print("# parameters:", sum(param.numel() for param in model.parameters()))
# model.load_state_dict(torch.load('./data7/model111.pt'))
# for m in model.modules():
#     if isinstance(m, nn.Conv1d):
#         nn.init.kaiming_normal_(m.weight)
#     elif isinstance(m, nn.LSTM):
#         for param in m.parameters():
#             if len(param.shape) >= 2:
#                 nn.init.orthogonal_(param.data)
#             else:
#                 nn.init.normal_(param.data)
#     elif isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, mean=0, std=np.sqrt(1/30))
# input = torch.rand(20, 1, 1024).to(device)
# with SummaryWriter(log_dir='logs', comment='model') as w:
#      w.add_graph(model, (input,))
# tb = program.TensorBoard()
# tb.configure(argv=[None, '--logdir', 'logs'])
# url = tb.launch()
# summary(model, input_size=(200, 200))  # 输出模型具有的参数
# criterion = nn.CrossEntropyLoss()
# criterion = GHMCC()
criterion = LSR()
#optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
#optimizer = optim.Adam(model.parameters(), lr=0.0006, weight_decay=0.01)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001, momentum=0.99)
bias_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias')
others_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias')
parameters = [{'parameters': bias_list, 'weight_decay': 0},
              {'parameters': others_list}]

optimizer = AdamP(model.parameters(), lr=0.0004, weight_decay=0.0001)#0.0002
# lr_scheduler =lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
# from Apollo import Apollo
# optimizer = Apollo(model.parameters(), lr=0.0002, weight_decay=0.0001)
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# from lookop import Lookahead
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0002, weight_decay=0.0001)
# optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)
# from Ranger2020 import Ranger
# optimizer = Ranger21(model.parameters(), lr=0.001, weight_decay=0, betas=(0.9, 0.999),
#                      use_warmup=True, normloss_factor=1e-4,
#                      use_adaptive_gradient_clipping=True,
#                      agc_clipping_value=.01, use_madgrad=False, use_warmdown=True, num_epochs=100,
#                      using_gc=True, num_batches_per_epoch=len(train_loader))
# optimizer = Ranger(model.parameters(), lr=0.001)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
losses = []
acces = []
eval_losses = []
eval_acces = []
early_stopping = EarlyStopping(patience=10, verbose=True)
starttime = time.time()
for epoch in range(150):  #150
    train_loss = 0
    train_acc = 0
    # print(optimizer.param_groups[0]['lr'])
    model.train()
    for img, label in train_loader:
        img = img.float()
        img = img.to(device)
        label = label.to(device)
        label = label.long()
        out = model(img)
        # 去掉out中维数是1的维度
        out = torch.squeeze(out).float()
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc
    # scheduler.step()
    # print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    # lr_scheduler.step()
    # lr_scheduler.step()
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0   # 将模型改为预测模式
    model.eval()
    for img, label in test_loader:
        img = img.type(torch.FloatTensor)
        img = img.to(device)
        label = label.to(device)
        label = label.long()
        # img = img.view(img.size(0), -1)
        out = model(img)
        out = torch.squeeze(out).float()
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc
    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    # print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
    print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
          .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader),
                  eval_loss / len(test_loader), eval_acc / len(test_loader)))
    early_stopping(eval_loss / len(test_loader), model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

endtime = time.time()
dtime = endtime - starttime
print("程序运行时间：%.8s s" % dtime)
torch.save(model.state_dict(), '\model111bn.pt')
import pandas as pd
pd.set_option('display.max_columns', None)   #显示完整的列
pd.set_option('display.max_rows', None)  #显示完整的行
