import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

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

def between_frame_loss(gait1, gait2, thres=0.01):
    g1 = gait1.permute(0, 2, 3, 1, 4).contiguous().view(gait1.shape[0], gait1.shape[2], gait1.shape[1]*gait1.shape[3])
    g2 = gait2.permute(0, 2, 3, 1, 4).contiguous().view(gait2.shape[0], gait2.shape[2], gait2.shape[1]*gait2.shape[3])
    num_batches = g1.shape[0]
    num_tsteps = g2.shape[1]
    mid_tstep = np.int(num_tsteps / 2) - 1
    loss = nn.functional.mse_loss(g1, g2)
    for bidx in range(num_batches):
        for tidx in range(num_tsteps):
            loss += nn.functional.mse_loss(g1[bidx, tidx, :]-g1[bidx, 0, :],
                                           g2[bidx, tidx, :]-g2[bidx, 0, :])
            loss += nn.functional.mse_loss(g1[bidx, tidx, :]-g1[bidx, mid_tstep, :],
                                           g2[bidx, tidx, :]-g2[bidx, mid_tstep, :])
            loss += nn.functional.mse_loss(g1[bidx, tidx, :]-g1[bidx, -1, :],
                                           g2[bidx, tidx, :]-g2[bidx, -1, :])
            for vidx in range(g1.shape[2]):
                if tidx > 0:
                    loss += nn.functional.mse_loss(g1[bidx, tidx, vidx] - g1[bidx, tidx-1, vidx],
                                                   g2[bidx, tidx, vidx] - g2[bidx, tidx-1, vidx])
                if tidx > 1:
                        loss += nn.functional.mse_loss(g1[bidx, tidx, vidx] -
                                                       2*g1[bidx, tidx-1, vidx] + g1[bidx, tidx-2, vidx],
                                                       g2[bidx, tidx, vidx] -
                                                       2 * g2[bidx, tidx - 1, vidx] + g2[bidx, tidx - 2, vidx])
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 5], g2[bidx, tidx-1, 5]+thres/2)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 6], g2[bidx, tidx-1, 6]+thres)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 7], g2[bidx, tidx-1, 7]+thres/3)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 8], g2[bidx, tidx-1, 8]+thres/2)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 9], g2[bidx, tidx-1, 9]+thres)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 10], g2[bidx, tidx-1, 10]+thres/3)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 11], g2[bidx, tidx-1, 11]+thres/2)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 12], g2[bidx, tidx-1, 12]+thres)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 13], g2[bidx, tidx-1, 13]+thres/3)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 14], g2[bidx, tidx-1, 14]+thres/2)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 15], g2[bidx, tidx-1, 15]+thres)
    return loss

def vae_loss(x_in, x_out, mean, lsig, beta=1.):
    # BCE = nn.functional.l1_loss(x_out, x_in)
    # BCE = nn.functional.binary_cross_entropy(x_out, x_in)
    # BCE = losses.affective_loss(x_in, x_out)
    BCE = between_frame_loss(x_in, x_out)
    KLD = -0.5 * torch.sum(1 + lsig - mean.pow(2) - lsig.exp())
    return BCE + beta*KLD