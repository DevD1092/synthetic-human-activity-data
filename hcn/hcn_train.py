from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import pickle
import argparse
import time
import pkbar
import glob
import pickle
import random
import scipy.misc
import cv2

from os import walk
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR, ExponentialLR
from eval_metric import calc_accuracy
from HCN_model import HCN,HCN2,HCN3,HCN4,ensemble_model,HCN1per_feat_1,HCN1per_256_fc9,HCN1per_feat256_fc9
from dataloader import NTURGBDData1,NTURGBDData1_classnum,NTURGBDDatasyn_classnum,NTURGBDData1_real_fid_classnum
from dataloader_integrate_act import NTURGBDData1_act

# Train: python hcn_train.py --temporal_length 60 --temporal_pattern interpolate --batch_size 64 --num_epochs 400 --expt clall_len60_per1_realper25 --num_person 1 --arch hcn --centering 1 --real_per 25 --mode train
# Test: python hcn_train.py --temporal_length 60 --temporal_pattern interpolate --batch_size 64 --num_epochs 400 --expt test_mode --num_person 1 --arch hcn --centering 1 --mode test --feat_map 1 --model_path /fast-stripe/workspaces/deval/synthetic-data/hcn/clall_len60_feat256_fc9_per1/model_state.pth --class_batch 0

# Global parameters for generating the Heatmpas
img_width,img_height=640,480  # Resolution of the video for which de-identified data was generated
heatmap_size = 224
target_size = 164
sigma = 10

def get_device(gpuid):
    if torch.cuda.is_available():
        device = 'cuda:%s'%str(gpuid)
    else:
        device = 'cpu'
    return device

def get_pickle(pickle_path):
    with open(pickle_path,'rb') as f:
        data_list = pickle.load(f)
    f.close()
    return data_list

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def init_seed(_):
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_keyps(keyps,img_width,img_height):
    non_zero_x = keyps[0,:,:,:] != 0
    non_zero_y = keyps[1,:,:,:] != 0
    keyps[0,non_zero_x] = (keyps[0,non_zero_x]-img_width/2)/(img_width)
    keyps[1,non_zero_y] = (keyps[1,non_zero_y]-img_height/2)/(img_height)
    return keyps

def normalize_keyps_vacnn(keyps,img_width,img_height,max_val,min_val):
    non_zero_x = keyps[0,:,:,:] != 0
    non_zero_y = keyps[1,:,:,:] != 0
    keyps[0,non_zero_x] = (keyps[0,non_zero_x]/img_width)*(max_val-min_val)/(min_val)
    keyps[1,non_zero_y] = (keyps[1,non_zero_y]/img_height)*(max_val-min_val)/(min_val)
    return keyps

def center(rgb):
    rgb[:,:,0] -= 110
    rgb[:,:,1] -= 110
    rgb[:,:,2] -= 110
    return rgb

def _transform(x):
    x = x.contiguous().view(x.size()[:2] + (-1, 3))

    rot = x.new(x.size()[0],3).uniform_(-0.3, 0.3)

    rot = rot.repeat(1, x.size()[1])
    rot = rot.contiguous().view((-1, x.size()[1], 3))
    rot = _rot(rot)
    x = torch.transpose(x, 2, 3)
    x = torch.matmul(rot, x)
    x = torch.transpose(x, 2, 3)

    x = x.contiguous().view(x.size()[:2] + (-1,))
    return x

def _rot(rot):
    cos_r, sin_r = rot.cos(), rot.sin()
    zeros = rot.new(rot.size()[:2] + (1,)).zero_()
    ones = rot.new(rot.size()[:2] + (1,)).fill_(1)

    r1 = torch.stack((ones, zeros, zeros),dim=-1)
    rx2 = torch.stack((zeros, cos_r[:,:,0:1], sin_r[:,:,0:1]), dim = -1)
    rx3 = torch.stack((zeros, -sin_r[:,:,0:1], cos_r[:,:,0:1]), dim = -1)
    rx = torch.cat((r1, rx2, rx3), dim = 2)

    ry1 = torch.stack((cos_r[:,:,1:2], zeros, -sin_r[:,:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,:,1:2], zeros, cos_r[:,:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 2)

    rz1 = torch.stack((cos_r[:,:,2:3], sin_r[:,:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,:,2:3], cos_r[:,:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 2)

    rot = rz.matmul(ry).matmul(rx)

    return rot

def reshape_data_vacnn(data):
    data = np.transpose(data,(1,3,2,0))
    data = np.reshape(data,(data.shape[0],data.shape[1],data.shape[2]*data.shape[3]))
    data = np.reshape(data,(data.shape[0],data.shape[1]*data.shape[2]))
    data = np.expand_dims(data,axis=0)
    return data

def convert_to_rgb_vacnn(data,max_val,min_val,cvm):
    zero_row = []
    cvm = int(cvm)
    ske_joint = np.squeeze(data,axis=0)
    #print (ske_joint.shape)
    for i in range(len(ske_joint)):
        if (ske_joint[i, :] == np.zeros((1, cvm))).all():
            zero_row.append(i)
    ske_joint = np.delete(ske_joint, zero_row, axis=0)
    
    if (ske_joint[:, 0:cvm//2] == np.zeros((ske_joint.shape[0], cvm//2))).all():
        ske_joint = np.delete(ske_joint, range(cvm//2), axis=1)
    elif (ske_joint[:, cvm//2:cvm] == np.zeros((ske_joint.shape[0], cvm//2))).all():
        ske_joint = np.delete(ske_joint, range(cvm//2, cvm), axis=1)
    
    # Convert to RGB
    ske_joint =  255 * (ske_joint - min_val) / (max_val - min_val)
    rgb_ske = np.reshape(ske_joint, (ske_joint.shape[0], ske_joint.shape[1] //3, 3))
    if rgb_ske.shape[0]==0:
        rgb_ske = np.zeros([224,224,3],dtype=np.float32)
    else:
        rgb_ske = scipy.misc.imresize(rgb_ske, (224, 224)).astype(np.float32)
    #rgb_ske = np.array(Image.fromarray(rgb_ske.astype('uint8')).resize((224,224))).astype(np.float32)
    rgb_ske = center(rgb_ske)
    rgb_ske = np.transpose(rgb_ske, [1, 0, 2])
    rgb_ske = np.transpose(rgb_ske, [2,1,0])
    return rgb_ske

def get_optical_flow(img_size,optical_flow_path):
    optical_flow_p1 = []
    optical_flow_p2 = []
    optical_flow_vid = get_pickle(optical_flow_path)

    for i in range(len(optical_flow_vid)):
        if np.all(np.array(optical_flow_vid[i][0]) == 0):
            optical_flow_p1.append(np.zeros((img_size,img_size,3),dtype=np.float32)) # Blank image for person optical flow
        else:
            optical_flow_arr = scipy.misc.imresize(optical_flow_vid[i][0], (img_size, img_size)).astype(np.float32)
            optical_flow_p1.append(optical_flow_arr)
        if np.all(np.array(optical_flow_vid[i][1]) == 0):
            optical_flow_p2.append(np.zeros((img_size,img_size,3),dtype=np.uint8)) # Blank image for person optical flow
        else:
            optical_flow_arr = scipy.misc.imresize(optical_flow_vid[i][1], (img_size, img_size)).astype(np.float32)
            optical_flow_p2.append(optical_flow_arr)
    return optical_flow_p1,optical_flow_p2

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# Transforms if needed
transform = transforms.Compose([transforms.ToTensor()])
transform_to_image = transforms.Compose([transforms.ToPILImage(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, ), (0.5, ))])

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def train(net,trainloader,testloader,num_epochs,lrate,expt,arch,milestones,temporal_length,model_path,model_path2,feat_map,class_num,mode):

    epoch_time=AverageMeter()
    model=net

    # Summary Writer Initiation
    default_dir = '/fast-stripe/workspaces/deval/synthetic-data/hcn/'
    ckpt_dir=os.path.join(default_dir,expt)
    create_dir(ckpt_dir)

    # Remove previous events for the same expt
    files = []
    for root,directory,filename in os.walk(ckpt_dir):
        files.extend(filename)
        break

    for filename in files:
        if "event" in filename:
            filepath=os.path.join(ckpt_dir,filename)
            os.remove(filepath)

    writer = SummaryWriter(ckpt_dir)

    # optimizer and scheduler
    LR_steps = milestones
    if mode == 'train':
        if arch == 'hcn' or arch == 'hcn_ensemble':
            optimizer = optim.Adam(model.parameters(),lr=lrate,betas=(0.9, 0.999),eps=1e-8,weight_decay=1e-4)
            #optimizer = optim.SGD(model.parameters(), lr=lrate, momentum=0.9, weight_decay=1e-4)
            scheduler = ExponentialLR(optimizer,gamma=0.99,last_epoch=-1)

    # Parameters to save in the dict
    train_loss=0
    train_acc=0
    best_train_epoch=0
    best_train_perf=0
    best_train_model=0
    best_test_acc=0
    best_test_model=0
    best_test_perf=0
    best_test_epoch=0

    for epoch in range(num_epochs):
        print ("Epoch",epoch+1, "/",num_epochs)
        batch_time=AverageMeter()
        start_epoch=time.time()

        losses=AverageMeter()
        accuracies = AverageMeter()
        test_acc = AverageMeter()
        if mode == 'train':
            pbar = pkbar.Pbar(name='Training...', target=len(trainloader))
            model.train()

            for i,data in enumerate(trainloader,0):
                start_batch = time.time()
                
                body_3d_keypoints = data[0].to(device)
                labels = data[1].long().to(device)

                if arch == 'hcn':
                    inputs = body_3d_keypoints

                optimizer.zero_grad()

                outputs = model(inputs)
                #print (outputs)
                #print (outputs.size())

                # Loss and gradient update
                criterion = nn.CrossEntropyLoss()
                train_loss = criterion(outputs, labels)
                losses.update(train_loss,inputs.size(0))
                train_loss.backward()
                optimizer.step()
                batch_time.update(time.time() - start_batch) 
                
                # Training accuracy
                train_acc = calc_accuracy(outputs,labels)
                accuracies.update(train_acc,outputs.size(0))

                pbar.update(i)
                lrate = optimizer.param_groups[0]['lr']
            #if arch == 'vacnn':
            #    scheduler.step(train_loss)
            #else:
            scheduler.step()

            # Train log - Summary Writer
            writer.add_scalar('Train/Loss',losses.avg,epoch)
            writer.add_scalar('Train/Acc', accuracies.avg, epoch)
            writer.add_scalar('Lrate', lrate, epoch)

            print ("Training accuracy -->",accuracies.avg)
            #print ("Training accuracy verify -->",np.mean(np.array(accuracies_verify)))
            print ("Epoch",epoch+1,"Learning rate ->",lrate)

            # Testing Loop
            state={
                'last_epoch': epoch+1,
                'best_train_state_dict': best_train_model,
                'last_state_dict': model.state_dict(),
                'best_train_perf': best_train_perf,
                'best_train_epoch':best_train_epoch,
                'best_test_perf': best_test_perf,
                'best_test_model':best_test_model,
                'best_test_epoch':best_test_epoch,
                'optimizer' : optimizer.state_dict(),
                'scheduler':scheduler
                }

            test_acc=test(testloader,state,None,model,arch,epoch,writer,test_acc,feat_map,ckpt_dir,class_num,mode)

            if accuracies.avg>best_train_perf:
                best_train_perf = accuracies.avg
                state['best_train_perf']=best_train_perf
                best_train_epoch=epoch+1
                state['best_train_epoch']=best_train_epoch
                best_train_model = model.state_dict()
                state['best_train_model']=best_train_model
            
            if test_acc.avg>best_test_acc:
                best_test_acc=test_acc.avg
                best_test_perf=test_acc.avg
                state['best_test_perf']=best_test_perf
                best_test_model=model.state_dict()
                state['best_test_model']=best_test_model
                best_test_epoch=epoch+1
                state['best_test_epoch']=best_test_epoch

            model_write_path = os.path.join(ckpt_dir,'model_state.pth')
            torch.save(state, model_write_path)

            epoch_time.update((time.time()-start_epoch))
            print ("Average epoch time ->",epoch_time.avg)
            print ("Best Train epoch ->",best_train_epoch,"Best Train Accuracy ->",best_train_perf)
            print ("Best Test epoch ->",best_test_epoch,"Best Test Accuracy ->",best_test_perf)
        
        elif mode == 'test':
            state = model_path
            state2 = model_path2
            test_acc = test(testloader,state,state2,model,arch,epoch,writer,test_acc,feat_map,ckpt_dir,class_num,mode)
        
    writer.close()

def test(testloader,state,state2,model,arch,epoch,writer,test_acc,feat_map,ckpt_dir,class_num,mode):
    test_losses = AverageMeter()
    if mode == 'train':
        state_dict=state['last_state_dict']
        model.load_state_dict(state_dict, strict=True)
        model.eval()
    elif mode == 'test':
        print ("Testing phase only")
        if arch == 'hcn':
            model.load_state_dict(torch.load(state)['best_test_model'], strict=False)
            print ("Loaded model",state,"| Best test per",torch.load(state)['best_test_perf'],"| Best test epoch",torch.load(state)['best_test_epoch'])
            model.eval()
        elif arch == 'hcn_ensemble':
            #print (state,state2)
            model[0].load_state_dict(torch.load(state)['best_test_model'],strict=False)
            print ("Loaded model",state,"| Best test per",torch.load(state)['best_test_perf'],"| Best test epoch",torch.load(state)['best_test_epoch'])
            model[1].load_state_dict(torch.load(state2)['best_test_model'],strict=False)
            print ("Loaded model",state2,"| Best test per",torch.load(state2)['best_test_perf'],"| Best test epoch",torch.load(state2)['best_test_epoch'])

    with torch.no_grad():
        feat_maps = []
        print ("Class",class_num)
        for i, data in enumerate(testloader,0):
            start_batch = time.time()

            body_3d_keypoints = data[0].to(device)
            labels = data[1].long().to(device)

            if arch == 'hcn' or 'hcn_ensemble':
                inputs = body_3d_keypoints

            if arch == 'hcn':
                outputs = model(inputs)
            elif arch == 'hcn_ensemble':
                outputs1 = model[0](inputs)
                outputs2 = model[1](inputs)
                #outputs = torch.max(outputs1,outputs2)
                outputs = outputs1 + outputs2

            if feat_map == 0:
                # Test loss
                criterion = nn.CrossEntropyLoss()
                test_loss = criterion(outputs, labels)
                test_losses.update(test_loss,inputs.size(0)) 
                
                # Testing accuracy
                acc = calc_accuracy(outputs,labels)
                test_acc.update(acc,outputs.size(0))
            elif feat_map == 1:
                #print (outputs)
                mean_feat_map = torch.mean(outputs,dim=0,keepdim=True)
                #print (mean_feat_map.shape)
                feat_maps.append(outputs)
                if i == 0:
                    outputs_sample = outputs.cpu().numpy()
                    #print (outputs_sample.shape)
                    feat_map_subsamples_dir = os.path.join(ckpt_dir,str(class_num))
                    create_dir(feat_map_subsamples_dir)
                    feat_map_subsamples_file = os.path.join(feat_map_subsamples_dir,'feat256_map_32.npy')
                    np.save(feat_map_subsamples_file,outputs_sample)
            elif feat_map == 2:
                #print (outputs)
                mean_feat_map = torch.mean(outputs,dim=0,keepdim=True)
                #print (mean_feat_map.shape)
                #feat_maps.append(outputs)
                if i == 0:
                    outputs_sample = outputs.cpu().numpy()
                    #print (outputs_sample.shape)
                    feat_map_subsamples_dir = os.path.join(ckpt_dir,'synthetic/cgan',str(class_num))
                    create_dir(feat_map_subsamples_dir)
                    feat_map_subsamples_file = os.path.join(feat_map_subsamples_dir,'feat_map_32.npy')
                    np.save(feat_map_subsamples_file,outputs_sample)
        
        if feat_map == 1 or feat_map == 2:
            #print (len(feat_maps))
            result = torch.cat(feat_maps,dim=0)
            #print (result.shape)
            #print (result)
            mean_result = torch.mean(result,dim=0,keepdim=True)
            #print (mean_result.shape)

        if feat_map == 0:
            # Test log - Summary Writer
            print ("Average test accuracy ->", test_acc.avg)
            writer.add_scalar('Test/Loss',test_losses.avg,epoch)
            writer.add_scalar('Test/Acc', test_acc.avg, epoch)
            return test_acc
        elif feat_map == 1:
            feat_map_dir = os.path.join(ckpt_dir,str(class_num))
            create_dir(feat_map_dir)
            mean_result = mean_result.cpu().numpy()
            #print (mean_result.shape)
            feat_map_file = os.path.join(feat_map_dir,'feat256_map.npy')
            np.save(feat_map_file,mean_result)
            print ("Output written to feat_map file")
            return 0
        elif feat_map == 2:
            print ("Output written to the feat_map file")
            return 0

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--temporal_pattern',default='random',type=str,help='Temporal pattern type -> "seq" for sequential "random" for random')
    parser.add_argument('--temporal_length',default=60,type=int,help='Temporal length for training')
    parser.add_argument('--feature_type',default='raw_keyps',type=str,help='"pairwise_dist","heatmaps",action_stamps","optical_flow". Enter "raw_keyps" only if training only on raw keypoints')
    parser.add_argument('--batch_size',default=128,type=int,help='Batch size')
    parser.add_argument('--num_epochs',default=200,type=int,help='Number of epochs')
    parser.add_argument('--lr',default=0.001,type=float,help='Learning rate')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--arch',default='hcn',type=str,help='Network architecture type -> linear;gru;bilstm;hcn;resnet')
    parser.add_argument('--modality',default='body',type=str,help='Modaility type - body;all')
    parser.add_argument('--expt',default='check',type=str,help='check or final')
    parser.add_argument('--augment',default='False',type=str,help='Augmentation type')
    parser.add_argument('--split',default='sub',type=str,help='sub/cam split')
    parser.add_argument('--input_type',default='numpy',type=str,help='h5py or numpy')
    parser.add_argument('--mode',default='train',type=str,help='train/test mode')
    parser.add_argument('--dataset',default='hcn',type=str,help='hcn/dgnn/vacnn')
    parser.add_argument('--optic_size',default=12,type=int,help='optical_flow_image_size')
    parser.add_argument('--fusion',default='late',type=str,help='early/late')
    parser.add_argument("--centering",type=int,default=0,help="Center the data or not")
    parser.add_argument("--normalize",type=int,default=0,help="Normalize the data or not")
    parser.add_argument("--spherical",type=int,default=0,help="Spherical the data or not")
    parser.add_argument("--num_person",type=int,default=2,help="HCN num_person model")
    parser.add_argument("--syn_per",type=int,default=0,help="Amount of synthetic data to use")
    parser.add_argument("--only_syn",type=int,default=0,help="Whether to train only on synthetic data -- 0 or 1")
    parser.add_argument("--real_per",type=int,default=100,help="Amount of real data to use")
    parser.add_argument("--model_path",type=str,default=None,help="If test mode the model path")
    parser.add_argument("--model_path2",type=str,default=None,help="If ensemble -> model path 2")
    parser.add_argument("--feat_map",type=int,default=0,help="Feature map save or not for real data: 1 ; for synthetic data:2")
    parser.add_argument("--class_batch",type=int,default=-1,help="Class batch 0 to 11")
    parser.add_argument("--val_split",type=int,default=0,help="Train with val split")
    
    args=parser.parse_args()
    device = get_device(args.gpu)
    #os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    init_seed(0)

    if args.class_batch != -1:
        start_ind = args.class_batch*4
        end_ind = (args.class_batch+1)*4
        classes = [i for i in range(start_ind,end_ind)]
        if args.class_batch == 11:
            classes.append(48)
        #print (classes)
    else:
        classes = [42]

    transform = None
    if args.split == 'sub':
        if args.val_split == 0:
            if args.mode == 'train':
                train_data = NTURGBDData1(args.temporal_length,'interpolate',args.gpu,'hcn','sub',args.normalize,args.centering,args.spherical,args.syn_per,args.only_syn,args.real_per,'train',args.expt)
                test_data = NTURGBDData1(args.temporal_length,'interpolate',args.gpu,'hcn','sub',args.normalize,args.centering,args.spherical,0,0,100,'test',args.expt)
            elif args.mode == 'test':
                if args.feat_map == 0:
                    train_data = None
                    test_data = NTURGBDData1(args.temporal_length,'interpolate',args.gpu,'hcn','sub',args.normalize,args.centering,args.spherical,0,0,'test',args.expt)
        elif args.val_split == 1:
            if args.mode == 'train':
                train_data = NTURGBDData1_act(args.temporal_length,'interpolate',args.gpu,'hcn','sub',args.normalize,args.centering,args.spherical,args.syn_per,args.only_syn,'train',args.expt)
                test_data = NTURGBDData1_act(args.temporal_length,'interpolate',args.gpu,'hcn','sub',args.normalize,args.centering,args.spherical,0,0,'test',args.expt)
    elif args.split == 'cam':
        if args.mode == 'train':
            train_data = NTURGBDData1(args.temporal_length,'interpolate',0,'hcn','sub',args.normalize,args.centering,args.spherical,args.syn_per,args.only_syn,args.real_per,'train',args.expt)
            test_data = NTURGBDData1(args.temporal_length,'interpolate',0,'hcn','sub',args.normalize,args.centering,args.spherical,0,0,100,'test',args.expt)
        elif args.mode == 'test':
            train_data = None
            test_data = NTURGBDData1(args.temporal_length,'interpolate',0,'hcn','sub',args.normalize,args.centering,args.spherical,0,0,100,'test',args.expt)

    if args.arch == 'hcn':
        if args.expt == 'check':
            num_workers = 0
        else:
            num_workers = 4
    else:
        num_workers = 0
    
    #milestones=[50,100,150]
    milestones=[30,60]
    #milestones=[3,5,8]
    if args.mode == 'train':
        trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,num_workers=num_workers,pin_memory=True)
        testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True,num_workers=num_workers,pin_memory=True)
    elif args.mode == 'test':
        if args.feat_map == 0:
            trainloader = None
            testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True,num_workers=num_workers,pin_memory=True)
            args.num_epochs = 1

    if args.temporal_length == 25:
        fc7_dim = 2048
    elif args.temporal_length == 60:
        fc7_dim = 4096

    if args.feat_map == 0:
        if args.arch == 'hcn':
            if args.num_person == 1:
                net = HCN4(in_channel=3,num_person=1,num_class=49,fc7_dim=fc7_dim).to(device) # HCN4 - FC8 HCN with 512 as fc; HCN1per_256_fc9 - FC9 with 256&512 as FCs
            elif args.num_person == 2:
                net = HCN(in_channel=3,num_person=2,num_class=49,fc7_dim=fc7_dim).to(device)
        elif args.arch == 'hcn_ensemble':
                net = ensemble_model(arch=args.arch,num_person=args.num_person,num_classes=49,fc7_dim=fc7_dim,device=device)
    elif args.feat_map == 1 or args.feat_map == 2:
        if args.arch == 'hcn':
            if args.num_person == 1:
                net = HCN1per_feat256_fc9(in_channel=3,num_person=1,num_class=49,fc7_dim=fc7_dim).to(device) # HCN1per_feat_1 - FC8 with 512 feat maps; HCN1per_feat256_fc9 - 256 feature maps
        
    print(net)

    ## Feat map save for the real data
    if args.feat_map == 1:
        train_data = None
        trainloader = None

        for class_num in classes:
            print ("Class ",class_num)
            test_data = NTURGBDData1_real_fid_classnum(args.temporal_length,'interpolate',args.gpu,'hcn','sub',args.normalize,args.centering,args.spherical,class_num,'train',transform,args.expt)
            testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True,num_workers=num_workers,pin_memory=True)
            args.num_epochs = 1
            train(net,trainloader,testloader,args.num_epochs,args.lr,args.expt,args.arch,milestones,args.temporal_length,args.model_path,args.model_path2,args.feat_map,class_num,args.mode)

    ## Feat map save for the synthetic data
    elif args.feat_map == 2:

        train_data = None
        trainloader = None
        
        classes = [args.class_batch]

        for class_num in classes:
            print ("Class",class_num)
            test_data = NTURGBDDatasyn_classnum(args.temporal_length,'interpolate',args.gpu,'cgan','sub',args.normalize,args.centering,args.spherical,class_num,'train',transform,args.expt)
            testloader = DataLoader(test_data,batch_size=args.batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)
            args.num_epocsh = 1
            train(net,trainloader,testloader,args.num_epochs,args.lr,args.expt,args.arch,milestones,args.temporal_length,args.model_path,args.model_path2,args.feat_map,class_num,args.mode)

    elif args.feat_map == 0:

        train(net,trainloader,testloader,args.num_epochs,args.lr,args.expt,args.arch,milestones,args.temporal_length,args.model_path,args.model_path2,args.feat_map,-1,args.mode)