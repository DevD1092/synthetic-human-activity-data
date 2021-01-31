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
from dataloader import NTURGBDData,vis_3dske,NTURGBDData_full
#from model import Generator,Discriminator
from tqdm import tqdm

##Commands:
#Train : python cgan2.py --img_width 60 --img_height 25 --channels 3 --temporal_length 60 --temporal_pattern interpolate --num_per 1 --centering 0 --dataset vacnn --sample_interval 40000 --save_steps 500 --n_epochs 5000 --batch_size 64 --expt vacnn_img25_per1
#Test : python cgan2.py --mode test --img_width 60 --img_height 25 --channels 3 --temporal_length 60 --temporal_pattern interpolate --num_per 1 --class_batch 0 --eval_samples 2048 --model_path 

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--beta2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--n_classes', type=int, default=49, help='number of classes for dataset')
parser.add_argument("--img_width", type=int, default=25, help="image width dimension")
parser.add_argument("--img_height", type=int, default=25, help="image height dimension")
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
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
opt = parser.parse_args()
print(opt)

C,H,W = opt.channels, opt.img_height, opt.img_width

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

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight, 1.0, 0.02)
        torch.nn.init.constant(m.bias, 0.0)

class Generator(nn.Module):
    # initializers
    def __init__(self,latent_dim=100,img_shape=(3,60,25),n_classes=49):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.n_classes = n_classes


        self.fc1_1 = nn.Linear(self.latent_dim, 128)
        self.fc1_1_bn = nn.BatchNorm1d(128)
        self.fc1_2 = nn.Linear(self.n_classes, 128)
        self.fc1_2_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(256, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, int(np.prod(self.img_shape)))


    # forward method
    def forward(self, input, label):
        x = F.relu(self.fc1_1_bn(self.fc1_1(input)))
        y = F.relu(self.fc1_2_bn(self.fc1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = F.tanh(self.fc4(x))
        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self, img_shape=(3,60,25),n_classes=49):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.n_classes = n_classes

        self.fc1_1 = nn.Linear(int(np.prod(self.img_shape)), 128)
        self.fc1_2 = nn.Linear(self.n_classes, 128)
        self.fc2 = nn.Linear(256, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 1)


    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.fc1_1(input.view(input.size(0),-1)), 0.2)
        y = F.leaky_relu(self.fc1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        #print (x.shape)
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
        x = F.sigmoid(self.fc4(x))
        return x


if opt.mode == 'train':

    
    ckpt_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan2',opt.expt,'checkpoints')
    images_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan2',opt.expt,'images')
    videos_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan2',opt.expt,'videos')
    models_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan2',opt.expt,'models')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(videos_dir,exist_ok=True)
    os.makedirs(models_dir,exist_ok=True)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize Generator and discriminator
    generator = Generator(latent_dim=opt.latent_dim,img_shape=(C,W,H),n_classes=opt.n_classes)
    discriminator = Discriminator(img_shape=(C,W,H),n_classes=opt.n_classes)

    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

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
    else:
        traindata = NTURGBDData_full(0,None,0,opt.dataset,'sub',0,0,0,opt.real_per,opt.num_per,'train',opt.expt)
    dataloader = DataLoader(traindata,batch_size=opt.batch_size,shuffle=True,num_workers=opt.n_cpu,pin_memory=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    writer = SummaryWriter(ckpt_dir)

    batches_done=0
    for epoch in range(opt.n_epochs):
        generator_loss = AverageMeter()
        discriminator_loss = AverageMeter()
        for i, (imgs, labels) in enumerate(dataloader):

            Batch_Size = imgs.shape[0]
            N_Class = opt.n_classes
            # Adversarial ground truths
            valid = Variable(torch.ones(Batch_Size).cuda(), requires_grad=False)
            fake = Variable(torch.zeros(Batch_Size).cuda(), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(torch.FloatTensor).cuda())

            real_y = torch.zeros(Batch_Size, N_Class)
            real_y = Variable(real_y.scatter_(1, labels.view(Batch_Size, 1), 1).cuda())
            #y = Variable(y.cuda())

            # Sample noise and labels as generator input
            noise = Variable(torch.randn((Batch_Size, opt.latent_dim)).cuda())
            #noise = Variable(FloatTensor(np.random.normal(0, 1, (Batch_Size, opt.latent_dim))))
            gen_labels = (torch.rand(Batch_Size, 1) * N_Class).type(torch.LongTensor)
            gen_y = torch.zeros(Batch_Size, N_Class)
            gen_y = Variable(gen_y.scatter_(1, gen_labels.view(Batch_Size, 1), 1).cuda())

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(noise, gen_y)
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs,gen_y).squeeze(), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Loss for real images
            d_real_loss = adversarial_loss(discriminator(real_imgs, real_y).squeeze(), valid)
            # Loss for fake images
            #gen_imgs = generator(noise, gen_y)
            d_fake_loss = adversarial_loss(discriminator(gen_imgs.detach(),gen_y).squeeze(), fake)
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss)

            d_loss.backward()
            optimizer_D.step()


            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                                d_loss.data.cpu(), g_loss.data.cpu()))

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                noise = Variable(torch.randn((N_Class**2, opt.latent_dim)).cuda())
                #noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (N_Class**2, opt.latent_dim))).cuda())
                #fixed labels
                y_ = torch.LongTensor(np.array([num for num in range(N_Class)])).view(N_Class,1).expand(-1,N_Class).contiguous()
                y_fixed = torch.zeros(N_Class**2, N_Class)
                y_fixed = Variable(y_fixed.scatter_(1,y_.view(N_Class**2,1),1).cuda())

                gen_imgs = generator(noise, y_fixed).view(-1,C,W,H)

                save_image(gen_imgs.data, "%s/%d.png" % (images_dir,batches_done), nrow=N_Class, normalize=False)
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