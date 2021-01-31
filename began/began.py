import argparse
import os
import numpy as np
import math
import sys
import random
sys.path.append('../')
import subprocess

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from tensorboardX import SummaryWriter
from dataloader import NTURGBDData,vis_3dske
from model import Generator,Discriminator
from tqdm import tqdm

def init_seed(_):
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--img_width", type=int, default=25, help="image width dimension")
parser.add_argument("--img_height", type=int, default=25, help="image height dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="number of image channels")
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
parser.add_argument("--lambda_k",type=float,default=0.001,help="Lambda k hyperparamter")
parser.add_argument("--gamma",type=float,default=0.75,help="Gamma hyperparamter")
opt = parser.parse_args()
print(opt)

if opt.class_batch != -1:
    start_ind = opt.class_batch*4
    end_ind = (opt.class_batch+1)*4
    classes = [i for i in range(start_ind,end_ind)]
    if opt.class_batch == 11:
        classes.append(48)
    #print (classes)
else:
    classes = [42]

img_shape = (opt.channels, opt.img_width, opt.img_height)

ntu_singleper_classdist = {25: 672, 37: 672, 38: 672, 24: 672, 46: 672, 45: 672, 26: 672, 40: 672, 41: 671, 35: 671, 31: 671, 42: 671, 33: 671, 34: 671, 47: 671, 48: 671, 44: 671, 43: 670, 23: 670, 32: 670, 36: 670, 22: 670, 27: 669, 14: 669, 6: 669, 21: 669, 29: 669, 16: 669, 20: 669, 28: 669, 7: 668, 9: 668, 17: 668, 30: 668, 13: 668, 18: 668, 5: 668, 4: 667, 15: 667, 3: 667, 12: 667, 19: 667, 0: 666, 1: 666, 2: 665, 10: 664, 8: 663, 11: 660, 39: 660}

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

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

# ----------
#  Training
# ----------

if opt.mode == 'train':

    init_seed(0)

    for class_num in classes:

        ckpt_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/began',str(class_num),opt.expt,'checkpoints')
        images_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/began',str(class_num),opt.expt,'images')
        videos_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/began',str(class_num),opt.expt,'videos')
        models_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/began',str(class_num),opt.expt,'models')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(videos_dir,exist_ok=True)
        os.makedirs(models_dir,exist_ok=True)

        # Initialize generator and discriminator
        if opt.arch == 'mlp':
            generator = Generator(latent_dim=opt.latent_dim,img_shape=img_shape)
            discriminator = Discriminator(img_shape=img_shape)

        if cuda:
            generator.cuda()
            discriminator.cuda()

        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

        # Configure data loader
        if opt.expt == 'check':
            opt.n_cpu = 0
        if opt.temporal_pattern == 'interpolate':
            traindata = NTURGBDData(opt.temporal_length,'interpolate',0,'hcn','sub',opt.normalize,opt.centering,opt.spherical,class_num,opt.real_per,opt.num_per,'train',opt.expt)
        else:
            traindata = NTURGBDData(0,None,0,'hcn','sub',0,0,0,class_num,opt.real_per,opt.num_per,'train',opt.expt)
        dataloader = DataLoader(traindata,batch_size=opt.batch_size,shuffle=True,num_workers=opt.n_cpu,pin_memory=True)

        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # BEGAN hyper parameters
        gamma = opt.gamma
        lambda_k = opt.lambda_k
        k = 0.0

        writer = SummaryWriter(ckpt_dir)

        for epoch in range(opt.n_epochs):
            generator_loss = AverageMeter()
            discriminator_loss = AverageMeter()
            for i, (imgs, _) in enumerate(dataloader):

                # Configure input
                #print (imgs.shape[0])
                real_imgs = Variable(imgs.type(Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

                # Generate a batch of images
                gen_imgs = generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = torch.mean(torch.abs(discriminator(gen_imgs) - gen_imgs))

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                d_real = discriminator(real_imgs)
                d_fake = discriminator(gen_imgs.detach())

                d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
                d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
                d_loss = d_loss_real - k * d_loss_fake

                d_loss.backward()
                optimizer_D.step()

                # ----------------
                # Update weights
                # ----------------

                diff = torch.mean(gamma * d_loss_real - d_loss_fake)

                # Update weight term for fake samples
                k = k + lambda_k * diff.item()
                k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

                # Update convergence metric
                #M = (d_loss_real + torch.abs(diff)).data[0]

                # --------------
                # Log Progress
                # --------------

                #print(
                #    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f"
                #    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), M, k)
                #)
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f], k: %f"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), k)
                )

                batches_done = epoch * len(dataloader) + i
                if batches_done % opt.sample_interval == 0:
                    save_image(gen_imgs.data[:25], "%s/%d.png" % (images_dir,batches_done), nrow=5, normalize=False)
                    save_image(gen_imgs.data[:25], "%s/%d_norm.png" % (images_dir,batches_done), nrow=5, normalize=True)
                    create_video(gen_imgs,batches_done,5,videos_dir)

                state={
                    'last_epoch': epoch+1,
                    'last_state_dict_G': generator.state_dict(),
                    'optimizer_G' : optimizer_G.state_dict(),
                    'last_state_dict_D': discriminator.state_dict(),
                    'optimizer_D':optimizer_D.state_dict()
                    }

                if (epoch+1)%opt.save_steps == 0:
                    model_write_path = os.path.join(models_dir,str(epoch+1))
                    os.makedirs(model_write_path,exist_ok=True)
                    model_write_path = os.path.join(model_write_path,'model_state_%d.pth'%opt.real_per)
                    torch.save(state, model_write_path)
            
                generator_loss.update(g_loss,imgs.shape[0])
                discriminator_loss.update(d_loss,imgs.shape[0])
                lrate_D = optimizer_D.param_groups[0]['lr']
                lrate_G = optimizer_G.param_groups[0]['lr']
            
            # Train log - Summary Writer
            writer.add_scalar('Generator/Loss',generator_loss.avg,epoch)
            writer.add_scalar('Discriminator/Loss', discriminator_loss.avg, epoch)
            writer.add_scalar('Generator/lrate',lrate_G,epoch)
            writer.add_scalar('Discriminator/lrate',lrate_D,epoch)

        writer.close()