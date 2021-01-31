import argparse
import os
import numpy as np
import math
import sys
import random
sys.path.append('../')
import subprocess

import numpy as np
import torch.nn as nn
import torch.optim as optim

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
from model import Generator,Discriminator,Generator1,Discriminator1
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
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=49, help="number of classes for dataset")
parser.add_argument("--img_width", type=int, default=25, help="image width dimension")
parser.add_argument("--img_height", type=int, default=25, help="image height dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--expt", type=str, default='check', help="Name of the expt")
parser.add_argument("--temporal_length", type=int, default=60, help="Temporal length")
parser.add_argument("--temporal_pattern", type=str, default=None, help="Temporal pattern")
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
parser.add_argument("--gpu",type=int,default=0,help="Which GPU device to use")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

if opt.mode == 'train':

    # parameters
    sample_num = opt.n_classes ** 2

    init_seed(0)

    ckpt_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/acgan2',opt.expt,'checkpoints')
    images_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/acgan2',opt.expt,'images')
    videos_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/acgan2',opt.expt,'videos')
    models_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/acgan2',opt.expt,'models')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(videos_dir,exist_ok=True)
    os.makedirs(models_dir,exist_ok=True)

    # Configure data loader
    transform = None
    if opt.expt == 'check':
        opt.n_cpu = 0
    if opt.temporal_pattern == 'interpolate':
        traindata = NTURGBDData_full(opt.temporal_length,'interpolate',0,'hcn','sub',opt.normalize,opt.centering,opt.spherical,opt.real_per,opt.num_per,'train',transform,opt.expt)
    else:
        traindata = NTURGBDData_full(0,None,0,'hcn','sub',0,0,0,opt.real_per,opt.num_per,'train',opt.expt)
    dataloader = DataLoader(traindata,batch_size=opt.batch_size,shuffle=True,num_workers=opt.n_cpu,pin_memory=True)

    # networks init
    Generator1 = Generator1(input_dim=opt.latent_dim, output_dim=opt.n_channels, input_size=opt.img_width,class_num=opt.n_classes)
    Discriminator1 = Discriminator1(input_dim=opt.n_channels, output_dim=1, input_size=opt.img_width,class_num=opt.n_classes)
    Generator1_optimizer = optim.Adam(Generator1.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
    Discriminator1_optimizer = optim.Adam(Discriminator1.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

    if cuda:
        Generator1.cuda()
        Discriminator1.cuda()
        BCE_loss = nn.BCELoss().cuda()
        CE_loss = nn.CrossEntropyLoss().cuda()
    else:
        BCE_loss = nn.BCELoss()
        CE_loss = nn.CrossEntropyLoss()

    print('---------- Networks architecture -------------')
    print (Generator1)
    print (Discriminator1)
    print('-----------------------------------------------')

    # Initialize weights
    Generator1.apply(weights_init_normal)
    Discriminator1.apply(weights_init_normal)

    # fixed noise & condition
    sample_z_ = torch.zeros((sample_num, opt.latent_dim))
    for i in range(opt.n_classes):
        sample_z_[i*opt.n_classes] = torch.rand(1, opt.latent_dim)
        for j in range(1, opt.n_classes):
            sample_z_[i*opt.n_classes + j] = sample_z_[i*opt.n_classes]

    temp = torch.zeros((opt.n_classes, 1))
    for i in range(opt.n_classes):
        temp[i, 0] = i

    temp_y = torch.zeros((sample_num, 1))
    for i in range(opt.n_classes):
        temp_y[i*opt.n_classes: (i+1)*opt.n_classes] = temp

    sample_y_ = torch.zeros((sample_num, opt.n_classes)).scatter_(1, temp_y.type(torch.LongTensor), 1)
    if cuda:
        sample_z_, sample_y_ = sample_z_.cuda(), sample_y_.cuda()
    
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(opt.batch_size, 1), torch.zeros(opt.batch_size, 1)
        if cuda:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

    Discriminator1.train()
    print('training start!!')
    start_time = time.time()
    for epoch in range(opt.n_epochs):
        Generator1.train()
        epoch_start_time = time.time()
        for iter, (x_, y_) in enumerate(self.data_loader):
            if iter == self.data_loader.dataset.__len__() // opt.batch_size:
                break
            z_ = torch.rand((opt.batch_size, opt.latent_dim))
            y_vec_ = torch.zeros((opt.batch_size, opt.n_classes)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
            if cuda:
                x_, z_, y_vec_ = x_.cuda(), z_.cuda(), y_vec_.cuda()

            # update D network
            Discriminator1_optimizer.zero_grad()

            D_real, C_real = Discriminator1(x_)
            D_real_loss = BCE_loss(D_real, self.y_real_)
            C_real_loss = CE_loss(C_real, torch.max(y_vec_, 1)[1])

            G_ = Generator1(z_, y_vec_)
            D_fake, C_fake = Discriminator1(G_)
            D_fake_loss = BCE_loss(D_fake, self.y_fake_)
            C_fake_loss = CE_loss(C_fake, torch.max(y_vec_, 1)[1])

            D_loss = D_real_loss + C_real_loss + D_fake_loss + C_fake_loss
            self.train_hist['D_loss'].append(D_loss.item())

            D_loss.backward()
            Discriminator1_optimizer.step()

            # update G network
            Generator1_optimizer.zero_grad()

            G_ = Generator1(z_, y_vec_)
            D_fake, C_fake = Discriminator1(G_)

            G_loss = BCE_loss(D_fake, self.y_real_)
            C_fake_loss = CE_loss(C_fake, torch.max(y_vec_, 1)[1])

            G_loss += C_fake_loss
            self.train_hist['G_loss'].append(G_loss.item())

            G_loss.backward()
            Generator1_optimizer.step()

            if ((iter + 1) % 100) == 0:
                print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                        ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // opt.batch_size, D_loss.item(), G_loss.item()))

        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
        with torch.no_grad():
            self.visualize_results((epoch+1))

    self.train_hist['total_time'].append(time.time() - start_time)
    print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                    opt.n_epochs, self.train_hist['total_time'][0]))
    print("Training finish!... save training results")

    self.save()
    utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                opt.n_epochs)
    utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        Generator1.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        image_frame_dim = int(np.floor(np.sqrt(sample_num)))

        if fix:
            """ fixed noise """
            samples = Generator1(self.sample_z_, self.sample_y_)
        else:
            """ random noise """
            sample_y_ = torch.zeros(opt.batch_size, opt.n_classes).scatter_(1, torch.randint(0, opt.n_classes - 1, (opt.batch_size, 1)).type(torch.LongTensor), 1)
            sample_z_ = torch.rand((opt.batch_size, opt.latent_dim))
            if cuda:
                sample_z_, sample_y_ = sample_z_.cuda(), sample_y_.cuda()

            samples = Generator1(sample_z_, sample_y_)

        if cuda:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(Generator1.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(Discriminator1.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        Generator1.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        Discriminator1.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))