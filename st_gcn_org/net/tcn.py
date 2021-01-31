import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

global cnt
cnt=0

class Unit_brdc(nn.Module):
    def __init__(self, D_in, D_out, kernel_size, stride=1, dropout=0):

        super(Unit_brdc, self).__init__()
        self.bn = nn.BatchNorm1d(D_in)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.conv = nn.Conv1d(
            D_in,
            D_out,
            kernel_size=kernel_size,
            padding=int((kernel_size - 1) / 2),
            stride=stride)

        # weight initialization
        conv_init(self.conv)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv(x)
        return x

class Unit_brdc_transpose(nn.Module):
    def __init__(self, D_in, D_out, kernel_size, stride=1, dropout=0):

        super(Unit_brdc_transpose, self).__init__()
        self.bn = nn.BatchNorm1d(D_in)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.conv = nn.ConvTranspose1d(
            D_in,
            D_out,
            kernel_size=kernel_size,
            padding=int((kernel_size - 1) / 2),
            stride=stride)

        # weight initialization
        conv_init(self.conv)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv(x)
        return x


class TCN_unit(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=9, stride=1):
        super(TCN_unit, self).__init__()
        self.unit1_1 = Unit_brdc(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            dropout=0.5,
            stride=stride)

        if in_channel != out_channel:
            self.down1 = Unit_brdc(
                in_channel, out_channel, kernel_size=1, stride=stride)
        else:
            self.down1 = None

    def forward(self, x):
        x =self.unit1_1(x)\
                + (x if self.down1 is None else self.down1(x))
        return x

class TCN_unit_transpose(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=9, stride=1):
        super(TCN_unit_transpose, self).__init__()
        self.unit1_1 = Unit_brdc(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            dropout=0.5,
            stride=stride)

        if in_channel != out_channel:
            self.down1 = Unit_brdc(
                in_channel, out_channel, kernel_size=1, stride=stride)
        else:
            self.down1 = None

    def forward(self, x):
        x =self.unit1_1(x)\
                + (x if self.down1 is None else self.down1(x))
        return x


class Model(nn.Module):
    def __init__(self, channel, num_class, num_per=1, num_joints=13, use_data_bn=False):
        super(Model, self).__init__()
        self.num_class = num_class
        self.use_data_bn = use_data_bn
        self.data_bn = nn.BatchNorm1d(channel * num_per * num_joints)
        self.conv0 = nn.Conv1d(channel * num_per * num_joints, 64, kernel_size=9, padding=4)
        conv_init(self.conv0)

        self.unit1 = TCN_unit(64, 64)
        self.unit2 = TCN_unit(64, 64)
        self.unit3 = TCN_unit(64, 64)
        self.unit4 = TCN_unit(64, 128, stride=2)
        self.unit5 = TCN_unit(128, 128)
        self.unit6 = TCN_unit(128, 128)
        self.unit7 = TCN_unit(128, 256, stride=2)
        self.unit8 = TCN_unit(256, 256)
        self.unit9 = TCN_unit(256, 256)
        self.bn = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.fcn = nn.Conv1d(256, num_class, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        #print ("x", x.size())
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)

        if self.use_data_bn:
            x = self.data_bn(x)

        x = self.conv0(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = self.unit3(x)
        x = self.unit4(x)
        x = self.unit5(x)
        x = self.unit6(x)
        x = self.unit7(x)
        x = self.unit8(x)
        x = self.unit9(x)
        x = self.bn(x)
        x = self.relu(x)

        x = F.avg_pool1d(x, kernel_size=x.size()[2])

        x = self.fcn(x)
        x = x.view(-1, self.num_class)
        x = self.sig(x)

        return x

class Model_transpose(nn.Module):
    def __init__(self, channel, num_class, num_per=1, num_joints=13, use_data_bn=False):
        super(Model_transpose, self).__init__()
        self.num_class = num_class
        self.use_data_bn = use_data_bn
        self.data_bn = nn.BatchNorm1d(3 * num_per * num_joints)
        #self.conv0 = nn.Conv1d(channel * num_per * num_joints, 64, kernel_size=9, padding=4)
        self.fcn = nn.Conv1d(channel, 64, kernel_size=1)
        conv_init(self.fcn)

        self.unit1 = TCN_unit(64, 64)
        self.unit2 = TCN_unit(64, 64)
        self.unit3 = TCN_unit(64, 64)
        self.unit4 = TCN_unit(64, 128, stride=2)
        self.unit5 = TCN_unit(128, 128)
        self.unit6 = TCN_unit(128, 128)
        self.unit7 = TCN_unit(128, 256, stride=2)
        self.unit8 = TCN_unit(256, 256)
        self.unit9 = TCN_unit(256, 3 * num_per * num_joints)
        self.tanh = nn.Tanh()
        #self.bn = nn.BatchNorm1d(256)
        #self.relu = nn.ReLU()

    def forward(self, x, l, C, T, V, M):
        #print ("x", x.size())
        global cnt
        N = x.size()[0]

        # concat
        x = torch.cat((x, l), dim=1)
        x = x.unsqueeze(dim=2)

        if cnt==0:
            print ("X+l shape",x.size())
        
        #x = F.avg_pool1d(x, kernel_size=x.size()[2])

        x = x.repeat([1, 1, T*4])
        if cnt==0:
            print ("Repeat,",x.size())

        x = self.fcn(x)
        if cnt==0:
            print ("FCN",x.size())


        #x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)

        #x = self.conv0(x)
        #if cnt==0:
        #    print ("Conv0",x.size())
        x = self.unit1(x)
        if cnt==0:
            print ("Unit1",x.size())
        x = self.unit2(x)
        if cnt==0:
            print ("Unit2",x.size())
        x = self.unit3(x)
        if cnt==0:
            print ("Unit3",x.size())
        x = self.unit4(x)
        if cnt==0:
            print ("Unit4",x.size())
        x = self.unit5(x)
        if cnt==0:
            print ("Unit5",x.size())
        x = self.unit6(x)
        if cnt==0:
            print ("Unit6",x.size())
        x = self.unit7(x)
        if cnt==0:
            print ("Unit7",x.size())
        x = self.unit8(x)
        if cnt==0:
            print ("Unit8",x.size())
        x = self.unit9(x)
        if cnt==0:
            print ("Unit9",x.size())

        if self.use_data_bn:
            x = self.data_bn(x)

        if cnt==0:
            print ("Size after bn",x.size())

        #x = self.fcn(x)
        #x = x.view(-1, self.num_class)


        x = self.tanh(x)

        x = x.view(-1,C,V,T)
        x = x.permute(0,1,3,2)
        x = x.unsqueeze(dim=4)

        if cnt==0:
            print ("Final x",x.size())
            cnt=1

        return x

def conv_init(module):
    # he_normal
    n = module.out_channels
    for k in module.kernel_size:
        n *= k
    module.weight.data.normal_(0, math.sqrt(2. / n))