import argparse
import os
import numpy as np
import math
import sys
import random
sys.path.append('../')
sys.path.append('../hcn/')
import subprocess
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR, ExponentialLR

import torch.nn as nn
import torch.nn.functional as F
import torch

from tensorboardX import SummaryWriter
from dataloader import NTURGBDData,vis_3dske,NTURGBDData_full, NTURGBDData1
from model import Generator,Discriminator,Generator_cnn
from tqdm import tqdm
from losses import calc_accuracy
from HCN_model import HCN4,HCN,HCN2,HCN3,HCN1per_feat,HCN1per_feat_1 #HCN4 - single person
from eval_metric import calc_accuracy

##Commands:
#Train : python cgan_learning_new.py --img_width 60 --img_height 25 --channels 3 --temporal_length 60 --temporal_pattern interpolate --num_per 1 --centering 1 --dataset hcn --sample_interval 4000 --save_steps 100 --n_epochs 500 --batch_size 64 --lambda_gen 1.0 --gen_pretrain 1 --act_pretrain 0 --multi_act 0 --expt img25_actnopretrain_genpretrain_lambda1p0_per1_ep500
#Test : python cgan.py --mode test --img_width 60 --img_height 25 --channels 3 --temporal_length 60 --temporal_pattern interpolate --num_per 1 --class_batch 0 --eval_samples 2048 --model_path 

def init_seed(_):
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def reshape_data_vacnn(data):
    data = np.transpose(data,(0,3,2,1))
    data = np.reshape(data,(data.shape[0],data.shape[1],data.shape[2]*data.shape[3]))
    return data

def convert_to_rgb_interpolate(data,max_val,min_val):
    
    ske_joint = np.squeeze(data,axis=0)

    ske_joint =  255 * (ske_joint - min_val) / (max_val - min_val)
    rgb_ske = np.reshape(ske_joint, (ske_joint.shape[0], ske_joint.shape[1] //3, 3))
    rgb_ske = np.transpose(rgb_ske, [1, 0, 2])
    rgb_ske = np.transpose(rgb_ske, [2,1,0])
    return rgb_ske

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=49, help="number of classes for dataset")
parser.add_argument("--img_width", type=int, default=25, help="image width dimension")
parser.add_argument("--img_height", type=int, default=25, help="image height dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--expt", type=str, default='check', help="Name of the expt")
parser.add_argument("--temporal_length", type=int, default=60, help="Temporal length")
parser.add_argument("--temporal_pattern", type=str, default=None, help="Temporal pattern")
parser.add_argument("--lambda_gp", type=int,default=10,help="Value of lambda_gp")
parser.add_argument("--normalize", type=int,default=0,help="Normalize the data or not")
parser.add_argument("--centering",type=int,default=0,help="Center the data or not")
parser.add_argument("--spherical",type=int,default=0,help="Spherical coords or not")
parser.add_argument("--save_steps",type=int,default=250,help="Steps interval for model saving")
parser.add_argument("--arch",type=str,default='mlp',help="Type of arch for G&D -> mlp,cnn")
parser.add_argument("--mode",type=str,default='train',help="train or test mode")
parser.add_argument("--model_path",type=str,default=None,help="Model path for testing mode")
parser.add_argument("--eval_samples",type=int,default=5,help="Number of eval samples to be produced")
parser.add_argument("--class_batch",type=int,default=-1,help="Class batch = if -1 then class 42, otherwise batches of 4")
parser.add_argument("--real_per",type=int,default=100,help="Amount of real data to be used - 25,50,75,100")
parser.add_argument("--num_per",type=int,default=1,help="Number of person to be considered")
parser.add_argument("--transform",type=int,default=0,help="Whether to have transform or not")
parser.add_argument("--dataset",type=str,default='hcn',help="Dataset to use - hcn or vacnn")
parser.add_argument("--act_pretrain",type=int,default=0,help="Action classifier pretrained or not")
parser.add_argument("--gen_pretrain",type=int,default=0,help="Generator pretrained or not")
parser.add_argument("--disc_pretrain",type=int,default=0,help="Discriminator pretrained or not")
parser.add_argument("--lambda_gen",type=float,default=1.0,help="Lambda for generated images loss")
parser.add_argument("--multi_act",type=int,default=0,help="Multiple action classifiers")
parser.add_argument("--conf_thresh",type=float,default=0.55,help="Confidence threshold for generated images")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_width, opt.img_height)

cuda = True if torch.cuda.is_available() else False

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

def create_video(gen_imgs,batches_done,num,videos_dir):
    
    data_to_save = gen_imgs.data[:num]
    #print (data_to_save.shape)
    vid_save_dir = os.path.join(videos_dir,str(batches_done))
    os.makedirs(vid_save_dir, exist_ok=True)
    max_val,min_val = 5.18858098984,-5.28981208801

    for i in range(data_to_save.shape[0]):
        data_3d = data_to_save[i].cpu().detach().numpy()
        #print (data_3d.shape)
        #data_3d = denormalize_data(data_3d,max_val,min_val,-1)
        vis_3dske(data_3d,opt.temporal_length,vid_save_dir,i,opt.normalize,opt.spherical)

        # Create a video of the plot images
        os.chdir(vid_save_dir)
        if opt.mode == 'test':
            #subprocess.call([
            #    'ffmpeg', '-framerate', '5', '-i', 'sam%05d'%(i)+'im%03d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            #    'crf','23','vis_act%05d.mp4'%(i)])
            subprocess.call([
                'ffmpeg', '-framerate', '1', '-i', 'sam%05d'%(i)+'im%03d.png', '-r', '1', '-pix_fmt', 'yuv420p',
                'vis_act%05d.mp4'%(i)])
            #subprocess.call(['ffmpeg', '-framerate', '1', '-i', 'input', 'vis_act%05d_fps.mp4'%i,'-r', '5', 'vis_act%05d_fps.mp4'%i])

def sample_image(n_row, batches_done):
            """Saves a grid of generated digits ranging from 0 to n_classes"""
            # Sample noise
            z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
            # Get labels ranging from 0 to n_classes for n rows
            labels = np.array([num for _ in range(n_row) for num in range(n_row)])
            labels = Variable(LongTensor(labels))
            gen_imgs = generator(z, labels)
            save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

def test_hcn(testloader,state_dict,state2,model,arch,epoch,writer,test_acc):
    test_losses = AverageMeter()
    #state_dict=state['last_state_dict']
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(testloader,0):
            start_batch = time.time()

            body_3d_keypoints = data[0].cuda()
            labels = data[1].long().cuda()

            if arch == 'hcn' or 'hcn_ensemble':
                inputs = body_3d_keypoints
            if arch == 'hcn':
                outputs = model(inputs)
            
            # Test loss
            criterion = nn.CrossEntropyLoss()
            test_loss = criterion(outputs, labels)
            test_losses.update(test_loss,inputs.size(0)) 
            
            # Testing accuracy
            acc = calc_accuracy(outputs,labels)
            test_acc.update(acc,outputs.size(0))

    # Test log - Summary Writer
    print ("Average test accuracy ->", test_acc.avg)
    writer.add_scalar('Test/Loss',test_losses.avg,epoch)
    writer.add_scalar('Test/Acc', test_acc.avg, epoch)
    return test_acc

# ----------
#  Training
# ----------

if opt.mode == 'train':

    init_seed(0)

    ckpt_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_gen_learning_new/gen_eval',opt.expt,'checkpoints')
    images_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_gen_learning_new/gen_eval',opt.expt,'images')
    videos_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_gen_learning_new/gen_eval',opt.expt,'videos')
    models_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_gen_learning_new/gen_eval',opt.expt,'models')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(videos_dir,exist_ok=True)
    os.makedirs(models_dir,exist_ok=True)

    #Loss functions
    if opt.arch == 'mlp':
        adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    if opt.arch == 'mlp':
        generator = Generator(latent_dim = opt.latent_dim, img_shape=img_shape, n_classes=opt.n_classes)
        discriminator = Discriminator(img_shape = img_shape, n_classes=opt.n_classes)
    elif opt.arch == 'cnn':
        generator = Generator_cnn(latent_dim=opt.latent_dim,DIM=256,classes=opt.n_classes)

    # HCN pre-trained model
    if opt.temporal_length == 25:
        fc7_dim = 2048
    elif opt.temporal_length == 60:
        fc7_dim = 4096
    if opt.num_per == 1:
        hcn_model = HCN4(in_channel=3,num_person=1,num_class=49,fc7_dim=fc7_dim).cuda()
    elif opt.num_per == 2:
        hcn_model = HCN(in_channel=3,num_person=2,num_class=49,fc7_dim=fc7_dim).cuda() 
    
    if opt.act_pretrain == 1:
        #hcn_model_path = '/fast-stripe/workspaces/deval/synthetic-data/hcn/CCC_models/clall_len60_per1/model_state.pth'        
        hcn_model_path = '/fast-stripe/workspaces/deval/synthetic-data/hcn/int_act_val_split_len60_per1/model_state.pth'
        hcn_model.load_state_dict(torch.load(hcn_model_path)['best_test_model'], strict=False)
        print ("Loaded action model",hcn_model_path,"| Best test per",torch.load(hcn_model_path)['best_test_perf'],"| Best test epoch",torch.load(hcn_model_path)['best_test_epoch'])
    else:
        print ("Not using any pretrained action classifer, training from scratch")
    if opt.gen_pretrain == 1:
        gen_model_path = '/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_changelabels_per1_ep5k/models/5000/model_state_100.pth'
        generator.load_state_dict(torch.load(gen_model_path)['last_state_dict_G'], strict=True)
        print ("Loaded generator model",gen_model_path)
    else:
        print ("Not using any pretrained generator, training from scratch")
    if opt.disc_pretrain == 1:
        disc_model_path = '/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_changelabels_per1_ep5k/models/5000/model_state_100.pth'
        discriminator.load_state_dict(torch.load(disc_model_path)['last_state_dict_D'],strict=True)
        print ("Loaded discriminator model",disc_model_path)
    else:
        print ("Not using any pretrained discriminator, training from scratch")

    if opt.multi_act == 1:
        
        if opt.num_per == 1:
            hcn_model_dup = HCN4(in_channel=3,num_person=1,num_class=49,fc7_dim=fc7_dim).cuda()
        elif opt.num_per == 2:
            hcn_model_dup = HCN(in_channel=3,num_person=2,num_class=49,fc7_dim=fc7_dim).cuda() 

        if opt.act_pretrain == 1:
            #hcn_model_path = '/fast-stripe/workspaces/deval/synthetic-data/hcn/CCC_models/clall_len60_per1/model_state.pth'        
            hcn_model_path = '/fast-stripe/workspaces/deval/synthetic-data/hcn/int_act_val_split_len60_per1/model_state.pth'
            hcn_model_dup.load_state_dict(torch.load(hcn_model_path)['best_test_model'], strict=False)
            print ("Loaded action model",hcn_model_path,"| Best test per",torch.load(hcn_model_path)['best_test_perf'],"| Best test epoch",torch.load(hcn_model_path)['best_test_epoch'])

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Configure data loader
    transform = None
    if opt.centering == 0 and opt.num_per == 1:
        mean = [ 0.0070, -0.1629,  3.2254]
        std = [0.3702, 0.4619, 0.4990]
    if opt.transform == 1:
        transform = transforms.Compose([transforms.Normalize(mean=mean,std=std)])
    if opt.expt == 'check':
        opt.n_cpu = 0
    if opt.temporal_pattern == 'interpolate':
        traindata = NTURGBDData_full(opt.temporal_length,'interpolate',0,opt.dataset,'sub',opt.normalize,opt.centering,opt.spherical,opt.real_per,opt.num_per,'train',transform,opt.expt)
        test_data = NTURGBDData1(opt.temporal_length,'interpolate',0,'hcn','sub',opt.normalize,opt.centering,opt.spherical,0,0,100,'test',opt.expt)
    else:
        traindata = NTURGBDData_full(0,None,0,opt.dataset,'sub',0,0,0,opt.real_per,opt.num_per,'train',opt.expt)
    dataloader = DataLoader(traindata,batch_size=opt.batch_size,shuffle=True,num_workers=opt.n_cpu,pin_memory=True)
    testloader = DataLoader(test_data,batch_size=opt.batch_size,shuffle=True,num_workers=opt.n_cpu,pin_memory=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    scheduler_G = ExponentialLR(optimizer_G, gamma=0.99, last_epoch=-1)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    scheduler_D = ExponentialLR(optimizer_D, gamma=0.99, last_epoch=-1)
    optimizer_hcn = torch.optim.Adam(hcn_model.parameters(),lr=0.001,betas=(0.9, 0.999),eps=1e-8,weight_decay=1e-4)
    scheduler_hcn = ExponentialLR(optimizer_hcn,gamma=0.99,last_epoch=-1)
    if opt.multi_act == 1:
        optimizer_hcn_dup = torch.optim.Adam(hcn_model_dup.parameters(),lr=0.001,betas=(0.9, 0.999),eps=1e-8,weight_decay=1e-4)
        scheduler_hcn_dup = ExponentialLR(optimizer_hcn_dup,gamma=0.99,last_epoch=-1)
    # scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=30)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    writer = SummaryWriter(ckpt_dir)

    best_test_acc = 0
    last_ep_test_acc=0

    for epoch in range(opt.n_epochs):
        generator_loss = AverageMeter()
        real_imgs_loss = AverageMeter()
        gen_imgs_loss = AverageMeter()
        accuracies_gen = AverageMeter()
        accuracies_real = AverageMeter()
        test_acc = AverageMeter()
        toKeep_avg = AverageMeter()

        hcn_model.train()
        generator.train()
        discriminator.train()

        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]
            #print ("Batch size",batch_size)
            hcn_data = imgs
            hcn_data = hcn_data.cuda()
            #print ("HCN data",hcn_data.shape)

            imgs = imgs[:,:,:,:,0:1]
            #print ("Images",imgs.shape)
            imgs = np.squeeze(imgs)
            #print ("Images",imgs.shape)
            if batch_size == 1:
                imgs = torch.unsqueeze(imgs,0)
            #print ("Images",imgs.shape)

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            if opt.arch == 'cnn':
                real_imgs = real_imgs.permute(0,1,3,2)
                max_val,min_val = 5.826573,-4.9881773
                #real_imgs =  255 * (real_imgs - min_val) / (max_val - min_val)
                #real_imgs =  (real_imgs/255)
            if opt.arch == 'mlp':
                labels = Variable(labels.type(LongTensor))

            if opt.arch == 'cnn':
                real_y = torch.zeros(batch_size, opt.n_classes)
                real_y = real_y.scatter_(1, labels.view(batch_size, 1), 1).view(batch_size, opt.n_classes, 1, 1).contiguous()
                real_y = Variable(real_y.expand(-1, -1, opt.img_width, opt.img_height).cuda())

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise and labels as generator input
            #print ("Batch size",batch_size)
            if opt.arch == 'mlp':
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
                #gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
                gen_labels = labels
            elif opt.arch == 'cnn':
                noise = Variable(torch.randn((batch_size, opt.latent_dim,1,1)).cuda())
                gen_labels = (torch.rand(batch_size, 1) * opt.n_classes).type(torch.LongTensor)
                gen_y = torch.zeros(batch_size, opt.n_classes)
                gen_y = Variable(gen_y.scatter_(1, gen_labels.view(batch_size, 1), 1).view(batch_size, opt.n_classes,1,1).cuda())
                gen_y_for_D = gen_y.view(batch_size, opt.n_classes, 1, 1).contiguous().expand(-1, -1, opt.img_width, opt.img_height)

            # Generate a batch of images
            #print ("Latent dim",z.shape)
            #print ("Generator labels",gen_labels.shape)
            if opt.arch == 'mlp':
                gen_imgs = generator(z, gen_labels)
            elif opt.arch == 'cnn':
                gen_imgs = generator(noise, gen_y)
                max_val,min_val = 5.826573,-4.9881773
                #gen_imgs =  255 * (gen_imgs - min_val) / (max_val - min_val)
                #gen_imgs =   (gen_imgs/255)

            # Loss for real images
            if opt.arch == 'mlp':
                validity_real = discriminator(real_imgs, labels)
                #validity_real = discriminator(real_imgs)
                #print ("Discriminator Real images validity",validity_real)
                #print ("Discriminator Real images validity", "Mean",torch.mean(validity_real), "STD:",torch.std(validity_real))
                #probs_validity_real = F.sigmoid(validity_real)
                #print ("Discriminator Real images probs",probs_validity_real)
                #print ("Discriminator Real images probs", "Mean",torch.mean(probs_validity_real), "STD:",torch.std(probs_validity_real))
                d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            if opt.arch == 'mlp':
                gen_imgs_detach = gen_imgs.detach()
                validity_fake = discriminator(gen_imgs_detach, gen_labels)
                #validity_fake = discriminator(gen_imgs.detach())
                d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # ---------------------
            #  Train Generator
            # ---------------------

            optimizer_G.zero_grad()

            # Loss measures generator's ability to fool the discriminator
            if opt.arch == 'mlp':
                gen_imgs = generator(z, gen_labels)
                validity = discriminator(gen_imgs, gen_labels)
                #validity = discriminator(gen_imgs)
                #print ("Discriminator generated images validity",validity)
                #print ("Discriminator generated images validity", "Mean",torch.mean(validity), "STD:",torch.std(validity))
                probs_validity = F.sigmoid(validity)
                #print ("Discriminator generated images probs",probs_validity)
                #print ("Discriminator generated images probs", "Mean",torch.mean(probs_validity), "STD:",torch.std(probs_validity))
                g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # -------------------------
            #  Train Action Classifier
            # -------------------------

            #gen_imgs_detach = gen_imgs.detach() #TODO: Detach or not?
            generator.eval()
            gen_imgs = generator(z, gen_labels)

            #optimizer_hcn.zero_grad()

            # Use the action classifier model to classify the generated images and propogate the cross-entropy loss backwards
            if opt.num_per == 1:
                #real_imgs = np.expand_dims(real_imgs,axis=3)
                #gen_imgs = np.expand_dims(gen_imgs,axis=3)
                #sec_per = np.zeros((batch_size,3,opt.temporal_length,25,1),dtype=np.float32)
                #sec_per = torch.from_numpy(sec_per).cuda()
                #gen_imgs_detach = torch.unsqueeze(gen_imgs_detach,4)
                gen_imgs = torch.unsqueeze(gen_imgs,4)
                real_imgs = torch.unsqueeze(real_imgs,4)

            criterion = nn.CrossEntropyLoss()

            # Train on the real data
            outputs_real = hcn_model(hcn_data)
            #predictedLabels_real = torch.argmax(outputs_real, 1)
            #probs_real = F.softmax(outputs_real, dim=1)
            #mostLikelyProbs_real = np.asarray([probs_real[j, predictedLabels_real[j]].item() for  j in range(len(probs_real))])
            #print ("Most likely real probs",mostLikelyProbs_real)
            action_loss_real = criterion(outputs_real,labels)
            train_acc_real = calc_accuracy(outputs_real,labels)
            accuracies_real.update(train_acc_real,batch_size)
            #action_loss_real.backward()
            #optimizer_hcn.step()
            #optimizer_hcn.zero_grad()
            
            # Train on the generated data
            #outputs_gen = hcn_model(gen_imgs_detach)
            outputs_gen = hcn_model(gen_imgs)
            # get a tensor of the labels that are most likely according to model
            predictedLabels = torch.argmax(outputs_gen, 1)
            #print ("Output gen",outputs_gen)
            #print ("Argmax Labels",predictedLabels)
            # psuedo labeling threshold
            confidenceThresh = opt.conf_thresh
            probs = F.softmax(outputs_gen, dim=1)
            #print ("Probs",probs)
            mostLikelyProbs = np.asarray([probs[j, predictedLabels[j]].item() for  j in range(len(probs))])
            print ("Most likely gen probs",mostLikelyProbs)

            '''Discriminator validity to select'''
            validity = validity.cpu().detach().numpy()
            if batch_size != 1:
                validity = np.transpose(validity,(1,0)).squeeze()
            else:
                validity = np.transpose(validity,(1,0))
            print ("validity",validity)
            probs_validity = probs_validity.cpu().detach().numpy()
            probs_validity = np.transpose(probs_validity,(1,0)).squeeze()
            print ("Probs validity",probs_validity)

            '''To Keep select'''
            #toKeep = mostLikelyProbs >= confidenceThresh
            #toKeep = predictedLabels == labels
            toKeep = validity >= confidenceThresh
            #toKeep = probs_validity >= confidenceThresh

            #print ("To keep",toKeep)
            #print (outputs_gen[toKeep])
            #print (predictedLabels[toKeep])
            #print ("Original Labels",labels)
            if sum(toKeep) != 0:
                #fakeClassifierLoss = criterion(outputs_gen[toKeep], labels[toKeep]) * opt.lambda_gen
                fakeClassifierLoss = criterion(outputs_gen[toKeep], labels[toKeep])
                #fakeClassifierLoss.backward()
                #optimizer_hcn.step()
                train_acc_gen = calc_accuracy(outputs_gen[toKeep],labels[toKeep])
                accuracies_gen.update(train_acc_gen,sum(toKeep))
            else:
                #print ("Inside Lambda Gen=0")
                opt.lambda_gen = 0.0
                fakeClassifierLoss = 0.0
            #action_loss_gen = criterion(outputs_gen, labels)
            #train_acc_gen = calc_accuracy(outputs_gen,labels)
            #accuracies_gen.update(train_acc_gen,batch_size)
            #action_loss_gen.backward()      

            action_loss = action_loss_real+ opt.lambda_gen * fakeClassifierLoss
            #action_loss = action_loss_real
            #action_loss = action_loss_gen
            action_loss.backward()
            optimizer_hcn.step()

            ## Reset the gradients
            #optimizer_G.zero_grad()
            optimizer_hcn.zero_grad()
            #optimizer_D.zero_grad()
            
            if sum(toKeep) != 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [loss_real: %f] [loss_gen: %f] [gen_acc: %f] [real_acc: %f] [LR_GEN: %f] [LR_ACT: %f] [Test_acc: %f] [Best Test acc: %f] [To Keep: %d]"
                    % (epoch, opt.n_epochs, i, len(dataloader), action_loss_real.item(), fakeClassifierLoss.item(), train_acc_gen, train_acc_real,optimizer_G.param_groups[0]['lr'], optimizer_hcn.param_groups[0]['lr'],\
                        last_ep_test_acc, best_test_acc, sum(toKeep)) #action_loss_real.item(),fakeClassifierLoss.item()
                )
                action_loss = action_loss_real + fakeClassifierLoss
                action_loss_gen = fakeClassifierLoss
            else:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [loss_real: %f] [loss_gen: %f] [gen_acc: %f] [real_acc: %f] [LR_GEN: %f] [LR_ACT: %f] [Test_acc: %f] [Best Test acc: %f] [To Keep: %d]"
                    % (epoch, opt.n_epochs, i, len(dataloader), action_loss_real.item(), 1000, 1000, train_acc_real,optimizer_G.param_groups[0]['lr'], optimizer_hcn.param_groups[0]['lr'],\
                        last_ep_test_acc, best_test_acc, sum(toKeep)) #action_loss_real.item()
                )
                action_loss = action_loss_real
                action_loss_gen = 1000
            
            #print(
            #        "[Epoch %d/%d] [Batch %d/%d] [loss_real: %f] [real_acc: %f] [LR_GEN: %f] [LR_ACT: %f] [Test_acc: %f] [Best Test acc: %f]"
            #        % (epoch, opt.n_epochs, i, len(dataloader), action_loss_real.item(), train_acc_real,optimizer_G.param_groups[0]['lr'], optimizer_hcn.param_groups[0]['lr'],\
            #            last_ep_test_acc, best_test_acc)
            #    )

            batches_done = epoch * len(dataloader) + i
            '''
            if batches_done % opt.sample_interval == 0:
                """Saves a grid of generated digits ranging from 0 to n_classes"""
                generator.eval()
                if opt.arch == 'mlp':
                    # Sample noise
                    z = Variable(FloatTensor(np.random.normal(0, 1, (5 ** 2, opt.latent_dim))))
                    # Get labels ranging from 0 to n_classes for n rows
                    labels = np.array([num for _ in range(5) for num in range(5)])
                    labels = Variable(LongTensor(labels))
                    gen_imgs = generator(z, labels)
                elif opt.arch == 'cnn':
                    noise = Variable(torch.randn((5, opt.latent_dim,1,1)).cuda())
                    gen_labels = (torch.rand(5, 1) * opt.n_classes).type(torch.LongTensor)
                    gen_y = torch.zeros(5, opt.n_classes)
                    gen_y = Variable(gen_y.scatter_(1, gen_labels.view(5, 1), 1).view(5, opt.n_classes,1,1).cuda())
                    gen_imgs = generator(noise, gen_y)
                    gen_imgs = gen_imgs.permute(0,1,3,2)
                save_image(gen_imgs.data[:25], "%s/%d.png" % (images_dir,batches_done), nrow=3, normalize=False)
                save_image(gen_imgs.data[:25], "%s/%d_norm.png" % (images_dir,batches_done), nrow=3, normalize=True)
                create_video(gen_imgs,batches_done,5,videos_dir)'''

            state={
                'last_epoch': epoch+1,
                'last_state_dict_G': generator.state_dict(),
                'optimizer_G' : optimizer_G.state_dict(),
                'optimizer_hcn': optimizer_hcn.state_dict(),
                'state_dict_hcn': hcn_model.state_dict(),
                'last_state_dict_D':discriminator.state_dict(),
                'optimizer_D':optimizer_D.state_dict(),
                'best_test_acc':best_test_acc
                }

            if (epoch+1)%opt.save_steps == 0:
                model_write_path = os.path.join(models_dir,str(epoch+1))
                os.makedirs(model_write_path,exist_ok=True)
                model_write_path = os.path.join(model_write_path,'model_state_%d.pth'%opt.real_per)
                torch.save(state, model_write_path)
            
            generator_loss.update(action_loss,imgs.shape[0])
            real_imgs_loss.update(action_loss_real,imgs.shape[0])
            gen_imgs_loss.update(action_loss_gen,imgs.shape[0])
            lrate_G = optimizer_G.param_groups[0]['lr']
            lrate_hcn = optimizer_hcn.param_groups[0]['lr']
            toKeep_avg.update(sum(toKeep),imgs.shape[0])
        
        scheduler_hcn.step()
        scheduler_G.step()
        scheduler_D.step()
        #scheduler_hcn_dup.step()

        ## Testing the Action Classifier model
        test_acc=test_hcn(testloader,hcn_model.state_dict(),None,hcn_model,'hcn',epoch,writer,test_acc)
        last_ep_test_acc = test_acc.avg
        if test_acc.avg>best_test_acc:
            best_test_acc=test_acc.avg

        # Train log - Summary Writer
        writer.add_scalar('Generator/Loss',generator_loss.avg,epoch)
        writer.add_scalar('Real Imgs/Loss',real_imgs_loss.avg,epoch)
        writer.add_scalar('Gen Imgs/Loss',gen_imgs_loss.avg,epoch)
        writer.add_scalar('HCN acc on Real Imgs',accuracies_real.avg,epoch)
        writer.add_scalar('HCN acc on Gen Imgs',accuracies_gen.avg,epoch)
        writer.add_scalar('Generator/lrate',lrate_G,epoch)
        writer.add_scalar('HCN/lrate',lrate_hcn,epoch)
        writer.add_scalar('Test_acc',test_acc.avg,epoch)
        writer.add_scalar('ToKeep',toKeep_avg.avg,epoch)
    
    writer.close()

    ## Check whether remote editing works

if opt.mode == 'test':

    init_seed(0)

    # Initialize generator and discriminator
    generator = Generator(latent_dim = opt.latent_dim, img_shape=img_shape, n_classes=opt.n_classes)

    if cuda:
        generator.cuda()

    ## Model path
    #/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_center_nonorm_per1_ep5k/models/5000/model_state_100.pth
    #/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_per1_ep500/models/100/model_state_100.pth
    #/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_bceloss_sigmoid_per1_ep5k/models/5000/model_state_100.pth
    #/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_changelabels_per1_ep5k/models/5000/model_state_100.pth
    #/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_largemodel_changelabels_per1_5p5k/models/5000/model_state_100.pth
    #/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_mlpsixlayer_changelabels_per1_p5k/models/5000/model_state_100.pth

    if opt.class_batch == -1:
        exit()
    else:
        start_ind = opt.class_batch*4
        end_ind = (opt.class_batch+1)*4
        classes = [i for i in range(start_ind,end_ind)]
        if opt.class_batch == 11:
            classes.append(48)

    dict = torch.load(opt.model_path)
    state_dict = dict['last_state_dict_G']
    generator.load_state_dict(state_dict,strict=True)
    generator.eval()

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    for class_num in classes:
        print ("Class num: ",class_num)
        videos_dir = opt.model_path.replace('model_state_%d.pth'%opt.real_per,'eval_op')
        os.makedirs(videos_dir,exist_ok=True)

        # Sample noise as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (opt.eval_samples, opt.latent_dim))))
        labels = []
        for j in range(0,opt.eval_samples):
            labels.append(class_num)
        labels=np.array(labels)
        gen_labels = Variable(LongTensor(labels))
        #gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, opt.eval_samples)))
        #print (gen_labels)
        #print (z)
        
        gen_imgs = generator(z, gen_labels)
        #print (fake_imgs.shape)
        #create_video(gen_imgs,opt.class_batch,opt.eval_samples,videos_dir)
        #print (gen_imgs.shape)
        
        ## Save synthetic data as numpy array
        output_dir = os.path.join(videos_dir,str(class_num))
        #output_dir = videos_dir
        os.makedirs(output_dir,exist_ok=True)
        syn_data = np.zeros((opt.eval_samples,3,opt.temporal_length,25,2),dtype=np.float32)
        output_file = os.path.join(output_dir,'syn_%d_samples.npy'%opt.eval_samples)
        gen_imgs = gen_imgs.cpu().detach().numpy()

        if opt.num_per == 2:
            data_per1 = gen_imgs[:,:,:,0:25]
            data_per2 = gen_imgs[:,:,:,25:50]
            data_per1 = np.expand_dims(data_per1,axis=4)
            data_per2 = np.expand_dims(data_per2,axis=4)
            syn_data[:,:,:,:,0:1] = data_per1
            syn_data[:,:,:,:,1:2] = data_per2
        elif opt.num_per == 1:
            gen_imgs = np.expand_dims(gen_imgs,axis=4)
            syn_data[:,:,:,:,0:1] = gen_imgs

        np.save(output_file,syn_data)

        print ("Evaluation completed")