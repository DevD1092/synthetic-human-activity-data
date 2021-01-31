# encoding: utf-8

"""
@author: huguyuehuhu
@time: 18-4-16 下午6:51
Permission is given to modify the code, any problem please contact huguyuehuhu@gmail.com
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import hcn_utils
import torchvision
import os
import sys
global cnt
cnt=0
class HCN(nn.Module):
    '''
    Input shape:
    Input shape should be (N, C, T, V, M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints
      and M is the number of people.
    '''
    def __init__(self,
                 in_channel=3,
                 num_joint=25,
                 num_person=2,
                 out_channel=64,
                 window_size=64,
                 num_class = 60,
                 fc7_dim=4096
                 ):
        super(HCN, self).__init__()
        self.num_person = num_person
        self.num_class = num_class
        # position
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))
        # motion
        self.conv1m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2m = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3m = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))

        # concatenate motion & position
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2,padding=1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.fc7= nn.Sequential(
            nn.Linear(fc7_dim,256*2), # out_channel *2 for temporal length 30 and out_channel *4 for temporal length 60
            nn.ReLU(),
            nn.Dropout2d(p=0.5))
        self.fc8 = nn.Linear(256*2,num_class)

        # initial weight
        hcn_utils.initial_model_weight(layers = list(self.children()))
        print('weight initial finished!')


    def forward(self, x,target=None):
        global cnt
        if cnt==0:
            print('x',x.size())
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        motion = x[:,:,1::,:,:]-x[:,:,0:-1,:,:]
        if cnt==0:
            print('motion',motion.size())
        motion = motion.permute(0,1,4,2,3).contiguous().view(N,C*M,T-1,V)
        if cnt==0:
            print('motion',motion.size())
        motion = F.upsample(motion, size=(T,V), mode='bilinear',align_corners=False).contiguous().view(N,C,M,T,V).permute(0,1,3,4,2)
        if cnt==0:
            print('motion',motion.size())
        logits = []
        for i in range(self.num_person):
            # position
            # N0,C1,T2,V3 point-level
            out = self.conv1(x[:,:,:,:,i])
            if cnt == 0:
                print('conv1', out.size())
            #print ("Conv 1 size ->",out.size())
            out = self.conv2(out)
            if cnt == 0:
                print('conv2', out.size())
            #print ("Conv2 size ->",out.size())
            # N0,V1,T2,C3, global level
            out = out.permute(0,3,2,1).contiguous()
            if cnt == 0:
                print('out', out.size())
            out = self.conv3(out)
            if cnt == 0:
                print('conv3', out.size())
            #print ("Conv3 size ->",out.size())
            out_p = self.conv4(out)
            if cnt == 0:
                print('conv4', out.size())
            #print ("Conv 4 size ->",out_p.size())


            # motion
            # N0,T1,V2,C3 point-level
            out = self.conv1m(motion[:,:,:,:,i])
            if cnt == 0:
                print('conv1m', out.size())
            #print ("Conv 1m size ->",out.size())
            out = self.conv2m(out)
            if cnt == 0:
                print('conv2m', out.size())
            #print ("Conv 2m size ->", out.size())
            # N0,V1,T2,C3, global level
            out = out.permute(0, 3, 2, 1).contiguous()
            if cnt == 0:
                print('out', out.size())
            out = self.conv3m(out)
            if cnt == 0:
                print('conv3m', out.size())
            #print ("Conv 3m size ->",out.size())
            out_m = self.conv4m(out)
            if cnt == 0:
                print('conv4m', out.size())
            #print ("Conv 4m size ->", out_m.size())

            # concat
            out = torch.cat((out_p,out_m),dim=1)
            if cnt == 0:
                print('out', out.size())
            out = self.conv5(out)
            if cnt == 0:
                print('conv5', out.size())
            #print ("Conv 5 size ->",out.size())
            out = self.conv6(out)
            if cnt == 0:
                print('conv6', out.size())
            #print ("Conv 6 size ->",out.size())

            logits.append(out)

        # max out logits
        #print (logits)
        #print (len(logits))
        #print (logits[0].size())
        out = torch.max(logits[0],logits[1])
        if cnt == 0:
            print('out', out.size())
        #out=logits[0]
        out = out.view(out.size(0), -1)
        if cnt == 0:
            print('out', out.size())
        # print(out.size())
        out = self.fc7(out)
        if cnt == 0:
            print('fc7', out.size())
        out = self.fc8(out)
        if cnt == 0:
            print('fc8', out.size())
            cnt=1

        t = out
        # if np.isnan(t.abs().sum().detach().cpu().numpy()):
        #     sam = x[0,0,:,:,0].detach().cpu().numpy()
        #     for i in range(0,sam.shape[0]):
        #         print(sam[i,:])
        #     print(sam.shape)
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor

        return out

class HCN4(nn.Module):
    '''
    Input shape:
    Input shape should be (N, C, T, V, M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints
      and M is the number of people.
    '''
    def __init__(self,
                 in_channel=3,
                 num_joint=25,
                 num_person=2,
                 out_channel=64,
                 window_size=64,
                 num_class = 60,
                 fc7_dim=4096
                 ):
        super(HCN4, self).__init__()
        self.num_person = num_person
        self.num_class = num_class
        # position
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))
        # motion
        self.conv1m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2m = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3m = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))

        # concatenate motion & position
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2,padding=1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.fc7= nn.Sequential(
            nn.Linear(fc7_dim,256*2), # out_channel *2 for temporal length 30 and out_channel *4 for temporal length 60
            nn.ReLU(),
            nn.Dropout2d(p=0.5))
        self.fc8 = nn.Linear(256*2,num_class)

        # initial weight
        hcn_utils.initial_model_weight(layers = list(self.children()))
        print('weight initial finished!')


    def forward(self, x,target=None):
        global cnt
        if cnt==0:
            print('x',x.size())
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        motion = x[:,:,1::,:,:]-x[:,:,0:-1,:,:]
        if cnt==0:
            print('motion',motion.size())
        motion = motion.permute(0,1,4,2,3).contiguous().view(N,C*M,T-1,V)
        if cnt==0:
            print('motion',motion.size())
        motion = F.upsample(motion, size=(T,V), mode='bilinear',align_corners=False).contiguous().view(N,C,M,T,V).permute(0,1,3,4,2)
        if cnt==0:
            print('motion',motion.size())
        logits = []
        for i in range(self.num_person):
            # position
            # N0,C1,T2,V3 point-level
            out = self.conv1(x[:,:,:,:,i])
            if cnt == 0:
                print('conv1', out.size())
            #print ("Conv 1 size ->",out.size())
            out = self.conv2(out)
            if cnt == 0:
                print('conv2', out.size())
            #print ("Conv2 size ->",out.size())
            # N0,V1,T2,C3, global level
            out = out.permute(0,3,2,1).contiguous()
            if cnt == 0:
                print('out', out.size())
            out = self.conv3(out)
            if cnt == 0:
                print('conv3', out.size())
            #print ("Conv3 size ->",out.size())
            out_p = self.conv4(out)
            if cnt == 0:
                print('conv4', out.size())
            #print ("Conv 4 size ->",out_p.size())


            # motion
            # N0,T1,V2,C3 point-level
            out = self.conv1m(motion[:,:,:,:,i])
            if cnt == 0:
                print('conv1m', out.size())
            #print ("Conv 1m size ->",out.size())
            out = self.conv2m(out)
            if cnt == 0:
                print('conv2m', out.size())
            #print ("Conv 2m size ->", out.size())
            # N0,V1,T2,C3, global level
            out = out.permute(0, 3, 2, 1).contiguous()
            if cnt == 0:
                print('out', out.size())
            out = self.conv3m(out)
            if cnt == 0:
                print('conv3m', out.size())
            #print ("Conv 3m size ->",out.size())
            out_m = self.conv4m(out)
            if cnt == 0:
                print('conv4m', out.size())
            #print ("Conv 4m size ->", out_m.size())

            # concat
            out = torch.cat((out_p,out_m),dim=1)
            if cnt == 0:
                print('out', out.size())
            out = self.conv5(out)
            if cnt == 0:
                print('conv5', out.size())
            #print ("Conv 5 size ->",out.size())
            out = self.conv6(out)
            if cnt == 0:
                print('conv6', out.size())
            #print ("Conv 6 size ->",out.size())

            logits.append(out)

        # max out logits
        #print (logits)
        #print (len(logits))
        #print (logits[0].size())
        #out = torch.max(logits[0],logits[1])
        out=logits[0]
        if cnt == 0:
            print('out', out.size())
        out = out.view(out.size(0), -1)
        if cnt == 0:
            print('out', out.size())
        # print(out.size())
        out = self.fc7(out)
        if cnt == 0:
            print('fc7', out.size())
        out = self.fc8(out)
        if cnt == 0:
            print('fc8', out.size())
            cnt=1

        t = out
        # if np.isnan(t.abs().sum().detach().cpu().numpy()):
        #     sam = x[0,0,:,:,0].detach().cpu().numpy()
        #     for i in range(0,sam.shape[0]):
        #         print(sam[i,:])
        #     print(sam.shape)
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor

        return out

class HCN1per_feat(nn.Module):
    '''
    Input shape:
    Input shape should be (N, C, T, V, M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints
      and M is the number of people.
    '''
    def __init__(self,
                 in_channel=3,
                 num_joint=25,
                 num_person=2,
                 out_channel=64,
                 window_size=64,
                 num_class = 60,
                 fc7_dim=4096
                 ):
        super(HCN1per_feat, self).__init__()
        self.num_person = num_person
        self.num_class = num_class
        # position
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))
        # motion
        self.conv1m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2m = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3m = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))

        # concatenate motion & position
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2,padding=1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.fc7= nn.Sequential(
            nn.Linear(fc7_dim,256*2), # out_channel *2 for temporal length 30 and out_channel *4 for temporal length 60
            nn.ReLU(),
            nn.Dropout2d(p=0.5))
        self.fc8 = nn.Linear(256*2,num_class)

        # initial weight
        hcn_utils.initial_model_weight(layers = list(self.children()))
        print('weight initial finished!')
    
    def sub_forward(self, x, target=None):
        global cnt
        if cnt==0:
            print('x',x.size())
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        motion = x[:,:,1::,:,:]-x[:,:,0:-1,:,:]
        if cnt==0:
            print('motion',motion.size())
        motion = motion.permute(0,1,4,2,3).contiguous().view(N,C*M,T-1,V)
        if cnt==0:
            print('motion',motion.size())
        motion = F.upsample(motion, size=(T,V), mode='bilinear',align_corners=False).contiguous().view(N,C,M,T,V).permute(0,1,3,4,2)
        if cnt==0:
            print('motion',motion.size())
        logits = []
        for i in range(self.num_person):
            # position
            # N0,C1,T2,V3 point-level
            out = self.conv1(x[:,:,:,:,i])
            if cnt == 0:
                print('conv1', out.size())
            #print ("Conv 1 size ->",out.size())
            out = self.conv2(out)
            if cnt == 0:
                print('conv2', out.size())
            #print ("Conv2 size ->",out.size())
            # N0,V1,T2,C3, global level
            out = out.permute(0,3,2,1).contiguous()
            if cnt == 0:
                print('out', out.size())
            out = self.conv3(out)
            if cnt == 0:
                print('conv3', out.size())
            #print ("Conv3 size ->",out.size())
            out_p = self.conv4(out)
            if cnt == 0:
                print('conv4', out.size())
            #print ("Conv 4 size ->",out_p.size())


            # motion
            # N0,T1,V2,C3 point-level
            out = self.conv1m(motion[:,:,:,:,i])
            if cnt == 0:
                print('conv1m', out.size())
            #print ("Conv 1m size ->",out.size())
            out = self.conv2m(out)
            if cnt == 0:
                print('conv2m', out.size())
            #print ("Conv 2m size ->", out.size())
            # N0,V1,T2,C3, global level
            out = out.permute(0, 3, 2, 1).contiguous()
            if cnt == 0:
                print('out', out.size())
            out = self.conv3m(out)
            if cnt == 0:
                print('conv3m', out.size())
            #print ("Conv 3m size ->",out.size())
            out_m = self.conv4m(out)
            if cnt == 0:
                print('conv4m', out.size())
            #print ("Conv 4m size ->", out_m.size())

            # concat
            out = torch.cat((out_p,out_m),dim=1)
            if cnt == 0:
                print('out', out.size())
            out = self.conv5(out)
            if cnt == 0:
                print('conv5', out.size())
            #print ("Conv 5 size ->",out.size())
            out = self.conv6(out)
            if cnt == 0:
                print('conv6', out.size())
            #print ("Conv 6 size ->",out.size())

            logits.append(out)

        # max out logits
        #print (logits)
        #print (len(logits))
        #print (logits[0].size())
        #out = torch.max(logits[0],logits[1])
        out=logits[0]
        if cnt == 0:
            print('out', out.size())
        out = out.view(out.size(0), -1)
        if cnt == 0:
            print('out', out.size())
        # print(out.size())
        out = self.fc7(out)
        if cnt == 0:
            print('fc7', out.size())
        #out = self.fc8(out)
        #if cnt == 0:
        #    print('fc8', out.size())
        cnt=1

        t = out
        # if np.isnan(t.abs().sum().detach().cpu().numpy()):
        #     sam = x[0,0,:,:,0].detach().cpu().numpy()
        #     for i in range(0,sam.shape[0]):
        #         print(sam[i,:])
        #     print(sam.shape)
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor

        return out

    def forward(self, x1, x2, target=None):
        out1 = self.sub_forward(x1)
        out2 = self.sub_forward(x2)
        
        return out1,out2

class HCN1per_feat_1(nn.Module):
    '''
    Input shape:
    Input shape should be (N, C, T, V, M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints
      and M is the number of people.
    '''
    def __init__(self,
                 in_channel=3,
                 num_joint=25,
                 num_person=2,
                 out_channel=64,
                 window_size=64,
                 num_class = 60,
                 fc7_dim=4096
                 ):
        super(HCN1per_feat_1, self).__init__()
        self.num_person = num_person
        self.num_class = num_class
        # position
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))
        # motion
        self.conv1m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2m = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3m = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))

        # concatenate motion & position
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2,padding=1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.fc7= nn.Sequential(
            nn.Linear(fc7_dim,256*2), # out_channel *2 for temporal length 30 and out_channel *4 for temporal length 60
            nn.ReLU(),
            nn.Dropout2d(p=0.5))
        self.fc8 = nn.Linear(256*2,num_class)

        # initial weight
        hcn_utils.initial_model_weight(layers = list(self.children()))
        print('weight initial finished!')
    
    def sub_forward(self, x, target=None):
        global cnt
        if cnt==0:
            print('x',x.size())
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        motion = x[:,:,1::,:,:]-x[:,:,0:-1,:,:]
        if cnt==0:
            print('motion',motion.size())
        motion = motion.permute(0,1,4,2,3).contiguous().view(N,C*M,T-1,V)
        if cnt==0:
            print('motion',motion.size())
        motion = F.upsample(motion, size=(T,V), mode='bilinear',align_corners=False).contiguous().view(N,C,M,T,V).permute(0,1,3,4,2)
        if cnt==0:
            print('motion',motion.size())
        logits = []
        for i in range(self.num_person):
            # position
            # N0,C1,T2,V3 point-level
            out = self.conv1(x[:,:,:,:,i])
            if cnt == 0:
                print('conv1', out.size())
            #print ("Conv 1 size ->",out.size())
            out = self.conv2(out)
            if cnt == 0:
                print('conv2', out.size())
            #print ("Conv2 size ->",out.size())
            # N0,V1,T2,C3, global level
            out = out.permute(0,3,2,1).contiguous()
            if cnt == 0:
                print('out', out.size())
            out = self.conv3(out)
            if cnt == 0:
                print('conv3', out.size())
            #print ("Conv3 size ->",out.size())
            out_p = self.conv4(out)
            if cnt == 0:
                print('conv4', out.size())
            #print ("Conv 4 size ->",out_p.size())


            # motion
            # N0,T1,V2,C3 point-level
            out = self.conv1m(motion[:,:,:,:,i])
            if cnt == 0:
                print('conv1m', out.size())
            #print ("Conv 1m size ->",out.size())
            out = self.conv2m(out)
            if cnt == 0:
                print('conv2m', out.size())
            #print ("Conv 2m size ->", out.size())
            # N0,V1,T2,C3, global level
            out = out.permute(0, 3, 2, 1).contiguous()
            if cnt == 0:
                print('out', out.size())
            out = self.conv3m(out)
            if cnt == 0:
                print('conv3m', out.size())
            #print ("Conv 3m size ->",out.size())
            out_m = self.conv4m(out)
            if cnt == 0:
                print('conv4m', out.size())
            #print ("Conv 4m size ->", out_m.size())

            # concat
            out = torch.cat((out_p,out_m),dim=1)
            if cnt == 0:
                print('out', out.size())
            out = self.conv5(out)
            if cnt == 0:
                print('conv5', out.size())
            #print ("Conv 5 size ->",out.size())
            out = self.conv6(out)
            if cnt == 0:
                print('conv6', out.size())
            #print ("Conv 6 size ->",out.size())

            logits.append(out)

        # max out logits
        #print (logits)
        #print (len(logits))
        #print (logits[0].size())
        #out = torch.max(logits[0],logits[1])
        out=logits[0]
        if cnt == 0:
            print('out', out.size())
        out = out.view(out.size(0), -1)
        if cnt == 0:
            print('out', out.size())
        # print(out.size())
        out = self.fc7(out)
        if cnt == 0:
            print('fc7', out.size())
        #out = self.fc8(out)
        #if cnt == 0:
        #    print('fc8', out.size())
        cnt=1

        t = out
        # if np.isnan(t.abs().sum().detach().cpu().numpy()):
        #     sam = x[0,0,:,:,0].detach().cpu().numpy()
        #     for i in range(0,sam.shape[0]):
        #         print(sam[i,:])
        #     print(sam.shape)
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor

        return out

    def forward(self, x1,target=None):
        out1 = self.sub_forward(x1)
        #out2 = self.sub_forward(x2)
        
        return out1

class HCN1per_256_fc9(nn.Module):
    '''
    Input shape:
    Input shape should be (N, C, T, V, M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints
      and M is the number of people.
    '''
    def __init__(self,
                 in_channel=3,
                 num_joint=25,
                 num_person=2,
                 out_channel=64,
                 window_size=64,
                 num_class = 60,
                 fc7_dim=4096
                 ):
        super(HCN1per_256_fc9, self).__init__()
        self.num_person = num_person
        self.num_class = num_class
        # position
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))
        # motion
        self.conv1m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2m = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3m = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))

        # concatenate motion & position
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2,padding=1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.fc7= nn.Sequential(
            nn.Linear(fc7_dim,256), # out_channel *2 for temporal length 30 and out_channel *4 for temporal length 60
            nn.ReLU(),
            nn.Dropout2d(p=0.5))

        self.fc8= nn.Sequential(
            nn.Linear(256,256*2), # out_channel *2 for temporal length 30 and out_channel *4 for temporal length 60
            nn.ReLU(),
            nn.Dropout2d(p=0.5))

        self.fc9 = nn.Linear(256*2,num_class)

        # initial weight
        hcn_utils.initial_model_weight(layers = list(self.children()))
        print('weight initial finished!')
    
    def sub_forward(self, x, target=None):
        global cnt
        if cnt==0:
            print('x',x.size())
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        motion = x[:,:,1::,:,:]-x[:,:,0:-1,:,:]
        if cnt==0:
            print('motion',motion.size())
        motion = motion.permute(0,1,4,2,3).contiguous().view(N,C*M,T-1,V)
        if cnt==0:
            print('motion',motion.size())
        motion = F.upsample(motion, size=(T,V), mode='bilinear',align_corners=False).contiguous().view(N,C,M,T,V).permute(0,1,3,4,2)
        if cnt==0:
            print('motion',motion.size())
        logits = []
        for i in range(self.num_person):
            # position
            # N0,C1,T2,V3 point-level
            out = self.conv1(x[:,:,:,:,i])
            if cnt == 0:
                print('conv1', out.size())
            #print ("Conv 1 size ->",out.size())
            out = self.conv2(out)
            if cnt == 0:
                print('conv2', out.size())
            #print ("Conv2 size ->",out.size())
            # N0,V1,T2,C3, global level
            out = out.permute(0,3,2,1).contiguous()
            if cnt == 0:
                print('out', out.size())
            out = self.conv3(out)
            if cnt == 0:
                print('conv3', out.size())
            #print ("Conv3 size ->",out.size())
            out_p = self.conv4(out)
            if cnt == 0:
                print('conv4', out.size())
            #print ("Conv 4 size ->",out_p.size())


            # motion
            # N0,T1,V2,C3 point-level
            out = self.conv1m(motion[:,:,:,:,i])
            if cnt == 0:
                print('conv1m', out.size())
            #print ("Conv 1m size ->",out.size())
            out = self.conv2m(out)
            if cnt == 0:
                print('conv2m', out.size())
            #print ("Conv 2m size ->", out.size())
            # N0,V1,T2,C3, global level
            out = out.permute(0, 3, 2, 1).contiguous()
            if cnt == 0:
                print('out', out.size())
            out = self.conv3m(out)
            if cnt == 0:
                print('conv3m', out.size())
            #print ("Conv 3m size ->",out.size())
            out_m = self.conv4m(out)
            if cnt == 0:
                print('conv4m', out.size())
            #print ("Conv 4m size ->", out_m.size())

            # concat
            out = torch.cat((out_p,out_m),dim=1)
            if cnt == 0:
                print('out', out.size())
            out = self.conv5(out)
            if cnt == 0:
                print('conv5', out.size())
            #print ("Conv 5 size ->",out.size())
            out = self.conv6(out)
            if cnt == 0:
                print('conv6', out.size())
            #print ("Conv 6 size ->",out.size())

            logits.append(out)

        # max out logits
        #print (logits)
        #print (len(logits))
        #print (logits[0].size())
        #out = torch.max(logits[0],logits[1])
        out=logits[0]
        if cnt == 0:
            print('out', out.size())
        out = out.view(out.size(0), -1)
        if cnt == 0:
            print('out', out.size())
        # print(out.size())
        out = self.fc7(out)
        if cnt == 0:
            print('fc7', out.size())
        out = self.fc8(out)
        if cnt == 0:
            print('fc8', out.size())
        out = self.fc9(out)
        if cnt == 0:
            print('fc9', out.size())
        cnt=1

        t = out
        # if np.isnan(t.abs().sum().detach().cpu().numpy()):
        #     sam = x[0,0,:,:,0].detach().cpu().numpy()
        #     for i in range(0,sam.shape[0]):
        #         print(sam[i,:])
        #     print(sam.shape)
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor

        return out

    def forward(self, x1,target=None):
        out1 = self.sub_forward(x1)
        #out2 = self.sub_forward(x2)
        
        return out1

class HCN1per_feat256_fc9(nn.Module):
    '''
    Input shape:
    Input shape should be (N, C, T, V, M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints
      and M is the number of people.
    '''
    def __init__(self,
                 in_channel=3,
                 num_joint=25,
                 num_person=2,
                 out_channel=64,
                 window_size=64,
                 num_class = 60,
                 fc7_dim=4096
                 ):
        super(HCN1per_feat256_fc9, self).__init__()
        self.num_person = num_person
        self.num_class = num_class
        # position
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))
        # motion
        self.conv1m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2m = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3m = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))

        # concatenate motion & position
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2,padding=1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.fc7= nn.Sequential(
            nn.Linear(fc7_dim,256), # out_channel *2 for temporal length 30 and out_channel *4 for temporal length 60
            nn.ReLU(),
            nn.Dropout2d(p=0.5))

        self.fc8= nn.Sequential(
            nn.Linear(256,256*2), # out_channel *2 for temporal length 30 and out_channel *4 for temporal length 60
            nn.ReLU(),
            nn.Dropout2d(p=0.5))

        self.fc9 = nn.Linear(256*2,num_class)

        # initial weight
        hcn_utils.initial_model_weight(layers = list(self.children()))
        print('weight initial finished!')
    
    def sub_forward(self, x, target=None):
        global cnt
        if cnt==0:
            print('x',x.size())
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        motion = x[:,:,1::,:,:]-x[:,:,0:-1,:,:]
        if cnt==0:
            print('motion',motion.size())
        motion = motion.permute(0,1,4,2,3).contiguous().view(N,C*M,T-1,V)
        if cnt==0:
            print('motion',motion.size())
        motion = F.upsample(motion, size=(T,V), mode='bilinear',align_corners=False).contiguous().view(N,C,M,T,V).permute(0,1,3,4,2)
        if cnt==0:
            print('motion',motion.size())
        logits = []
        for i in range(self.num_person):
            # position
            # N0,C1,T2,V3 point-level
            out = self.conv1(x[:,:,:,:,i])
            if cnt == 0:
                print('conv1', out.size())
            #print ("Conv 1 size ->",out.size())
            out = self.conv2(out)
            if cnt == 0:
                print('conv2', out.size())
            #print ("Conv2 size ->",out.size())
            # N0,V1,T2,C3, global level
            out = out.permute(0,3,2,1).contiguous()
            if cnt == 0:
                print('out', out.size())
            out = self.conv3(out)
            if cnt == 0:
                print('conv3', out.size())
            #print ("Conv3 size ->",out.size())
            out_p = self.conv4(out)
            if cnt == 0:
                print('conv4', out.size())
            #print ("Conv 4 size ->",out_p.size())


            # motion
            # N0,T1,V2,C3 point-level
            out = self.conv1m(motion[:,:,:,:,i])
            if cnt == 0:
                print('conv1m', out.size())
            #print ("Conv 1m size ->",out.size())
            out = self.conv2m(out)
            if cnt == 0:
                print('conv2m', out.size())
            #print ("Conv 2m size ->", out.size())
            # N0,V1,T2,C3, global level
            out = out.permute(0, 3, 2, 1).contiguous()
            if cnt == 0:
                print('out', out.size())
            out = self.conv3m(out)
            if cnt == 0:
                print('conv3m', out.size())
            #print ("Conv 3m size ->",out.size())
            out_m = self.conv4m(out)
            if cnt == 0:
                print('conv4m', out.size())
            #print ("Conv 4m size ->", out_m.size())

            # concat
            out = torch.cat((out_p,out_m),dim=1)
            if cnt == 0:
                print('out', out.size())
            out = self.conv5(out)
            if cnt == 0:
                print('conv5', out.size())
            #print ("Conv 5 size ->",out.size())
            out = self.conv6(out)
            if cnt == 0:
                print('conv6', out.size())
            #print ("Conv 6 size ->",out.size())

            logits.append(out)

        # max out logits
        #print (logits)
        #print (len(logits))
        #print (logits[0].size())
        #out = torch.max(logits[0],logits[1])
        out=logits[0]
        if cnt == 0:
            print('out', out.size())
        out = out.view(out.size(0), -1)
        if cnt == 0:
            print('out', out.size())
        # print(out.size())
        out = self.fc7(out)
        if cnt == 0:
            print('fc7', out.size())
        #out = self.fc8(out)
        #if cnt == 0:
        #    print('fc8', out.size())
        #out = self.fc9(out)
        #if cnt == 0:
        #    print('fc9', out.size())
        cnt=1

        t = out
        # if np.isnan(t.abs().sum().detach().cpu().numpy()):
        #     sam = x[0,0,:,:,0].detach().cpu().numpy()
        #     for i in range(0,sam.shape[0]):
        #         print(sam[i,:])
        #     print(sam.shape)
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor

        return out

    def forward(self, x1,target=None):
        out1 = self.sub_forward(x1)
        #out2 = self.sub_forward(x2)
        
        return out1

class HCN2(nn.Module):
    '''
    Input shape:
    Input shape should be (N, C, T, V, M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints
      and M is the number of people.
    '''
    def __init__(self,
                 in_channel=3,
                 num_joint=25,
                 num_person=2,
                 out_channel=64,
                 window_size=64,
                 num_class = 60,
                 fc7_dim=4096
                 ):
        super(HCN2, self).__init__()
        self.num_person = num_person
        self.num_class = num_class
        # position
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))
        # motion
        self.conv1m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2m = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3m = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))

        # concatenate motion & position
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2,padding=1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        # self.fc7= nn.Sequential(
        #     nn.Linear(fc7_dim,256*2), # out_channel *2 for temporal length 30 and out_channel *4 for temporal length 60
        #     nn.ReLU(),
        #     nn.Dropout2d(p=0.5))
        # self.fc8 = nn.Linear(256*2,num_class)

        # initial weight
        hcn_utils.initial_model_weight(layers = list(self.children()))
        print('weight initial finished!')


    def forward(self, x,target=None):
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        global cnt
        if cnt==0:
            print('x',x.size())
        motion = x[:,:,1::,:,:]-x[:,:,0:-1,:,:]
        if cnt==0:
            print('motion',motion.size())
        motion = motion.permute(0,1,4,2,3).contiguous().view(N,C*M,T-1,V)
        motion = F.upsample(motion, size=(T,V), mode='bilinear',align_corners=False).contiguous().view(N,C,M,T,V).permute(0,1,3,4,2)
        if cnt==0:
            print('motion',motion.size())
        logits = []
        for i in range(self.num_person):
            # position
            # N0,C1,T2,V3 point-level
            out = self.conv1(x[:,:,:,:,i])
            if cnt == 0:
                print('conv1', out.size())
            #print ("Conv 1 size ->",out.size())
            out = self.conv2(out)
            if cnt == 0:
                print('conv2', out.size())
            #print ("Conv2 size ->",out.size())
            # N0,V1,T2,C3, global level
            out = out.permute(0,3,2,1).contiguous()
            out = self.conv3(out)
            if cnt == 0:
                print('conv3', out.size())
            #print ("Conv3 size ->",out.size())
            out_p = self.conv4(out)
            if cnt == 0:
                print('conv4', out.size())
            #print ("Conv 4 size ->",out_p.size())


            # motion
            # N0,T1,V2,C3 point-level
            out = self.conv1m(motion[:,:,:,:,i])
            if cnt == 0:
                print('conv1m', out.size())
            #print ("Conv 1m size ->",out.size())
            out = self.conv2m(out)
            if cnt == 0:
                print('conv2m', out.size())
            #print ("Conv 2m size ->", out.size())
            # N0,V1,T2,C3, global level
            out = out.permute(0, 3, 2, 1).contiguous()
            out = self.conv3m(out)
            if cnt == 0:
                print('conv3m', out.size())
            #print ("Conv 3m size ->",out.size())
            out_m = self.conv4m(out)
            if cnt == 0:
                print('conv4m', out.size())
            #print ("Conv 4m size ->", out_m.size())

            # concat
            out = torch.cat((out_p,out_m),dim=1)
            out = self.conv5(out)
            if cnt == 0:
                print('conv5', out.size())
            #print ("Conv 5 size ->",out.size())
            out = self.conv6(out)
            if cnt == 0:
                print('conv6', out.size())
            #print ("Conv 6 size ->",out.size())

            logits.append(out)

        # max out logits
        #print (logits)
        #print (len(logits))
        #print (logits[0].size())
        out = torch.max(logits[0],logits[1])
        #out=logits[0]
        out = out.view(out.size(0), -1)
        if cnt == 0:
            print('out', out.size())
            cnt=1
        # print(out.size())
        # out = self.fc7(out)
        # out = self.fc8(out)

        t = out
        # print(t.abs().sum())
        assert not ((t != t).any())# find out nan in tensor
        # assert not (t.abs().sum() == 0) # find out 0 tensor

        return out

class HCN3(nn.Module):
    '''
    Input shape:
    Input shape should be (N, C, T, V, M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints
      and M is the number of people.
    '''
    def __init__(self,
                 in_channel=3,
                 num_joint=25,
                 num_person=2,
                 out_channel=64,
                 window_size=64,
                 num_class = 60,
                 fc7_dim=4096
                 ):
        super(HCN3, self).__init__()
        self.num_person = num_person
        self.num_class = num_class
        # position
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))
        # motion
        self.conv1m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2m = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3m = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))

        # concatenate motion & position
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2,padding=1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.fc7= nn.Sequential(
            nn.Linear(fc7_dim,256*2), # out_channel *2 for temporal length 30 and out_channel *4 for temporal length 60
            nn.ReLU(),
            nn.Dropout2d(p=0.5))
        self.fc8 = nn.Linear(256*2,num_class)

        # initial weight
        hcn_utils.initial_model_weight(layers = list(self.children()))
        print('weight initial finished!')


    def forward(self, x,x1,target=None):
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        motion = x[:,:,1::,:,:]-x[:,:,0:-1,:,:]
        motion = motion.permute(0,1,4,2,3).contiguous().view(N,C*M,T-1,V)
        motion = F.upsample(motion, size=(T,V), mode='bilinear',align_corners=False).contiguous().view(N,C,M,T,V).permute(0,1,3,4,2)

        N, C, T, V, M = x1.size()  # N0, C1, T2, V3, M4
        motion_x1 = x1[:,:,1::,:,:]-x1[:,:,0:-1,:,:]
        motion_x1 = motion_x1.permute(0,1,4,2,3).contiguous().view(N,C*M,T-1,V)
        motion_x1 = F.upsample(motion_x1, size=(T,V), mode='bilinear',align_corners=False).contiguous().view(N,C,M,T,V).permute(0,1,3,4,2)

        logits = []
        for i in range(self.num_person):
            # position
            # N0,C1,T2,V3 point-level
            out = self.conv1(x[:,:,:,:,i])
            #print ("Conv 1 size ->",out.size())
            out = self.conv2(out)
            #print ("Conv2 size ->",out.size())
            # N0,V1,T2,C3, global level
            out = out.permute(0,3,2,1).contiguous()
            out = self.conv3(out)
            #print ("Conv3 size ->",out.size())
            out_p = self.conv4(out)
            #print ("Conv 4 size ->",out_p.size())


            # motion
            # N0,T1,V2,C3 point-level
            out = self.conv1m(motion[:,:,:,:,i])
            #print ("Conv 1m size ->",out.size())
            out = self.conv2m(out)
            #print ("Conv 2m size ->", out.size())
            # N0,V1,T2,C3, global level
            out = out.permute(0, 3, 2, 1).contiguous()
            out = self.conv3m(out)
            #print ("Conv 3m size ->",out.size())
            out_m = self.conv4m(out)
            #print ("Conv 4m size ->", out_m.size())

            # concat
            out = torch.cat((out_p,out_m),dim=1)
            out = self.conv5(out)
            #print ("Conv 5 size ->",out.size())
            out = self.conv6(out)
            #print ("Conv 6 size ->",out.size())

            logits.append(out)

        # max out logits
        #print (logits)
        #print (len(logits))
        #print (logits[0].size())
        out = torch.max(logits[0],logits[1])
        #out=logits[0]
        out = out.view(out.size(0), -1)
        # print(out.size())
        # out = self.fc7(out)
        # out = self.fc8(out)

        ## Bone input
        logits_x1 = []
        for i in range(self.num_person):
            # position
            # N0,C1,T2,V3 point-level
            out_x1 = self.conv1(x1[:,:,:,:,i])
            #print ("Conv 1 size ->",out.size())
            out_x1 = self.conv2(out_x1)
            #print ("Conv2 size ->",out.size())
            # N0,V1,T2,C3, global level
            out_x1 = out_x1.permute(0,3,2,1).contiguous()
            out_x1 = self.conv3(out_x1)
            #print ("Conv3 size ->",out.size())
            out_p_x1 = self.conv4(out_x1)
            #print ("Conv 4 size ->",out_p.size())


            # motion
            # N0,T1,V2,C3 point-level
            out_x1 = self.conv1m(motion_x1[:,:,:,:,i])
            #print ("Conv 1m size ->",out.size())
            out_x1 = self.conv2m(out_x1)
            #print ("Conv 2m size ->", out.size())
            # N0,V1,T2,C3, global level
            out_x1 = out_x1.permute(0, 3, 2, 1).contiguous()
            out_x1 = self.conv3m(out_x1)
            #print ("Conv 3m size ->",out.size())
            out_m_x1 = self.conv4m(out_x1)
            #print ("Conv 4m size ->", out_m.size())

            # concat
            out_x1 = torch.cat((out_p_x1,out_m_x1),dim=1)
            out_x1 = self.conv5(out_x1)
            #print ("Conv 5 size ->",out.size())
            out_x1 = self.conv6(out_x1)
            #print ("Conv 6 size ->",out.size())

            logits_x1.append(out_x1)

        # max out logits
        #print (logits)
        #print (len(logits))
        #print (logits[0].size())
        out_x1 = torch.max(logits_x1[0],logits_x1[1])
        #out=logits[0]
        out_x1 = out_x1.view(out_x1.size(0), -1)
        # print(out.size())
        # out = self.fc7(out)
        # out = self.fc8(out)

        out_add = torch.add(out,out_x1)

        out = self.fc7(out_add)
        out = self.fc8(out)

        t = out
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor

        return out

class ensemble(nn.Module):
    def __init__(self, models, fc7_dim = 4096):
        super(ensemble, self).__init__()
        self.models = len(models)
        self.networks = nn.ModuleList()

        self.fc7 = nn.Sequential(
                nn.Linear(fc7_dim, 256 * 2),
                # out_channel *2 for temporal length 30 and out_channel *4 for temporal length 60
                nn.ReLU(),
                nn.Dropout2d(p=0.5))
       
        self.fc8 = nn.Linear(256 * 2,60)

        for i in range(0, len(models)):
            self.networks.append(models[i])
    
    def forward(self,x1):
        map1 = self.networks[0](x1) # Real model
        map2 = self.networks[1](x1) # Synthetic model
        
        map1 = self.fc7(map1)
        map2 = self.fc7(map2)

        out = map1 + map2
        
        #out = torch.cat((map1,map2),dim=-1)
        #out = self.fc7(out)
        out = self.fc8(out)

        t = out
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor

        return out

def ensemble_model(arch,num_person,num_classes,fc7_dim,device):
    models = []

    if arch == 'hcn_ensemble':
        if num_person == 1:
            model = HCN4(in_channel=3,num_person=num_person,num_class=num_classes,fc7_dim=fc7_dim).to(device)
            models.append(model)
            models.append(model)
        elif num_person == 2:
            model = HCN(in_channel=3,num_person=num_person,num_class=num_classes,fc7_dim=fc7_dim).to(device)
            models.append(model)
            models.append(model)

    return models

def loss_fn(outputs,labels,current_epoch=None,params=None):
    """
    Compute the cross entropy loss given outputs and labels.
    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    if params.loss_args["type"] == 'CE':
        CE = nn.CrossEntropyLoss()(outputs, labels)
        loss_all = CE
        loss_bag = {'ls_all': loss_all, 'ls_CE': CE}
    #elif: other losses

    return loss_bag


def accuracytop1(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

def accuracytop2(output, target, topk=(2,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

def accuracytop3(output, target, topk=(3,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

def accuracytop5(output, target, topk=(5,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracytop1': accuracytop1,
    'accuracytop5': accuracytop5,
    # could add more metrics such as accuracy for each token type
}

if __name__ == '__main__':
    model = HCN()
    children = list(model.children())
    print(children)
