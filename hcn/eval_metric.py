from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np

def calc_accuracy(outputs,labels):
    '''Computes the true positives; Pass output and labels tensor to this function'''
    correct=0
    batch_size = outputs.size(0)

    label_cpu = labels.detach().cpu().numpy()
    outputs_cpu = F.log_softmax(outputs,dim=1).detach().cpu().numpy()
    pred = np.argmax(outputs_cpu,axis=1)
    correct += (pred == label_cpu).sum()
    accuracy = 100*(correct/batch_size)
    return accuracy
