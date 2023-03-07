# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:05:08 2021

@author: Administrator
"""
import numpy as np
import torch
import os
import re
import scipy.io as scio
from torch.utils import data as da
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#########数据载入模块################
raw_num = 400
class Data(object):
    '''
    读取mat格式数据，由于每个故障数据数量不同，这里只截取前480000个数据
    get_data()产生的数据为（2400，2000）的输入数据
    get_label()产生的数据为（2400，1）的标签数据
    '''

    def __init__(self):
        self.data = self.get_data()
        #获取数据
        self.label = self.get_label()
        #获取数据标签
    def file_list(self):
        return os.listdir('./pppudata/')
    #返回该目录下的所有文件夹
    def get_data(self):
        file_list = self.file_list()
        x = np.zeros((1024, 0))
        #data3为了提高样本的数量，每组选取400组，一共1200组数据，其中测试集组数为360组，
        #相关参数为   400，160223，1022.
        #print(file_list)
        for i in range(len(file_list)):
            file = scio.loadmat('./pppudata/{}'.format(file_list[i]))
            for k in file.keys():
                file_matched = re.match('data', k)
                #file_matched = re.match('unnamed', k)
                if file_matched:
                    key = file_matched.group()
            #if i == 0:
            print(np.array(file[key].shape))
            data1 = np.array(file[key][0:409600])
            for j in range(0, len(data1)-1023, 1024): #
                  x = np.concatenate((x, data1[j:j+1024]), axis=1)
            #else:
                #data = np.hstack((data, file[key][0:10]))
        return x.T
    def get_label(self):
        file_list = self.file_list()
        title = np.array([i.replace('.mat', '') for i in file_list])
        #字典解析式   file_list是一个字典，遍历这个字典。将i中包含.mat的字段用‘’来替换,replace在python中只能用来替换字符串
        #用空格来替换掉.mat这个文件格式
        label = title[:, np.newaxis]
        #将一维数组转为一维矩阵，方便处理
        label_copy = np.copy(label)
        for _ in range(raw_num-1):
            label = np.hstack((label, label_copy))
            #生成标签数据，横向堆叠很多次，一定要记得转换成数组
        return label.flatten()
Data = Data()
data = Data.data
# import matplotlib.pyplot as plt
# data1 = data[401, :].squeeze()
# x = np.linspace(0, 1024, num=1024, endpoint=True, retstep=False, dtype=None)
# x = x.squeeze()
# plt.plot(x, data1)
# font2 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 14.5,
#          }
# plt.xticks(fontsize=14.5, weight='normal', family='Times New Roman')
# plt.yticks(fontsize=14.5, weight='normal', family='Times New Roman')
# plt.xlabel('sampling points', font2)
# plt.ylabel('Amplitude(m/$\mathregular{s^2}$)', font2)
# plt.tick_params(labelsize=14.5)
# plt.savefig('t-sne1', format='tif', dpi=200)
# plt.show()
label = Data.label
y = label.astype("int32")
# # lb = LabelBinarizer()
# # y = lb.fit_transform(label)   #标签二值化
# # dataNew = "G:\\研究生资料\\研二\\实践code(pytorch)\\datanew.mat"
# # scio.savemat(dataNew,mdict={'data':data,'label':label})
# ##############################数据归一化处理###########################
#ss = MinMaxScaler()
ss = StandardScaler()
data = data.T
data = ss.fit_transform(data).T
############################划分训练集和测试集#######################

#X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=2, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=2, stratify=y)
X_train = torch.from_numpy(X_train).unsqueeze(1)
X_test = torch.from_numpy(X_test).unsqueeze(1)

# y_train = torch.from_numpy(y_train)
# y_test = torch.from_numpy(y_test)
class TrainDataset(da.Dataset):
    def __init__(self):
        self.Data = X_train
        self.Label = y_train
    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label
    def __len__(self):
        return len(self.Data)
class TestDataset(da.Dataset):
    def __init__(self):
        self.Data = X_test
        self.Label = y_test
    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label
    def __len__(self):
        return len(self.Data)
Train = TrainDataset()
Test = TestDataset()
train_loader = da.DataLoader(Train, batch_size=16, shuffle=True)#batch_size可以直接设置，用于调整参数。
#for img, label in train_loader:
 #   print(img.shape,'\n',label.shape)--(100,1,1024)--(100)
#由于train_test_split已经打乱顺序，这里不需要继续打乱顺序
test_loader = da.DataLoader(Test, batch_size=10, shuffle=False)
# test_loader  = da.DataLoader(Train, batch_size=100, shuffle=False)#batch_size可以直接设置，用于调整参数。
# #由于train_test_split已经打乱顺序，这里不需要继续打乱顺序
# train_loader = da.DataLoader(Test, batch_size=50, shuffle=True)
