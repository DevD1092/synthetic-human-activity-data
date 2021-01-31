import argparse
import os
import numpy as np
import math
import sys
import random
sys.path.append('../')
sys.path.append('../st_gcn_vae')
sys.path.append('../st_gcn_org')
import subprocess

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
from dataloader import NTURGBDData,vis_3dske,NTURGBDData_full
from model import Generator,Discriminator,Generator_cnn,Discriminator_cnn
from st_gcn_vae import gcn_model
#from st_gcn_org.net.st_gcn import Model as Encoder_org
from st_gcn_org.net.st_gcn_org import Model as Encoder_org
from st_gcn_org.net.st_gcn import Decoder as Decoder_org
from gcn_model import Encoder, Decoder
from st_gcn_org.net.tcn import Model as Encoder_tcn
from st_gcn_org.net.tcn import Model_transpose as Decoder_tcn
from tqdm import tqdm
from torchviz import make_dot

##Commands:
#Train : python cgan.py --img_width 60 --img_height 25 --channels 3 --temporal_length 60 --temporal_pattern interpolate --num_per 1 --centering 1 --dataset hcn --sample_interval 4000 --save_steps 100 --n_epochs 500 --batch_size 64 --real_per 25 --arch cnn --latent_dim 512 --expt realper25_z512_cnn_nosig_mseloss_linearblock_per1_ep500
#Test : python cgan.py --mode test --img_width 60 --img_height 25 --channels 3 --temporal_length 60 --temporal_pattern interpolate --num_per 1 --class_batch 0 --eval_samples 2048 --arch cnn --latent_dim 512 --real_per 10 --model_path 

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

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def conv_init(module):
    # he_normal
    n = module.out_channels
    for k in module.kernel_size:
        n *= k
    module.weight.data.normal_(0, math.sqrt(2. / n))

def init_weights(m):
    if type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

# ----------
#  Training
# ----------

if opt.mode == 'train':

    init_seed(0)

    if opt.arch == 'gcn':
        ckpt_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_gcn',opt.expt,'checkpoints')
        images_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_gcn',opt.expt,'images')
        videos_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_gcn',opt.expt,'videos')
        models_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_gcn',opt.expt,'models')
    elif opt.arch == 'gcn_org':
        ckpt_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_gcn_org',opt.expt,'checkpoints')
        images_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_gcn_org',opt.expt,'images')
        videos_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_gcn_org',opt.expt,'videos')
        models_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_gcn_org',opt.expt,'models')
    elif opt.arch == 'tcn':
        ckpt_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_tcn',opt.expt,'checkpoints')
        images_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_tcn',opt.expt,'images')
        videos_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_tcn',opt.expt,'videos')
        models_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_tcn',opt.expt,'models')
    else:
        ckpt_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan',opt.arch,opt.expt,'checkpoints')
        images_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan',opt.arch,opt.expt,'images')
        videos_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan',opt.arch,opt.expt,'videos')
        models_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan',opt.arch,opt.expt,'models')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(videos_dir,exist_ok=True)
    os.makedirs(models_dir,exist_ok=True)

    # Loss functions
    if opt.arch == 'mlp':
        adversarial_loss = torch.nn.MSELoss()
    elif opt.arch == 'cnn':
        #adversarial_loss = torch.nn.BCELoss()
        #adversarial_loss = torch.nn.BCEWithLogitsLoss()
        adversarial_loss = torch.nn.MSELoss()
    elif opt.arch == 'gcn' or opt.arch == 'gcn_org' or opt.arch == 'tcn':
        #adversarial_loss = torch.nn.BCELoss()
        #adversarial_loss = torch.nn.BCEWithLogitsLoss()
        adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    if opt.arch == 'mlp':
        generator = Generator(latent_dim = opt.latent_dim, img_shape=img_shape, n_classes=opt.n_classes)
        discriminator = Discriminator(img_shape = img_shape, n_classes=opt.n_classes)
    elif opt.arch == 'cnn':
        generator = Generator_cnn(latent_dim=opt.latent_dim,DIM=256,classes=opt.n_classes)
        discriminator = Discriminator_cnn(d=256)
        generator.apply(init_weights)
        discriminator.apply(init_weights)
    elif opt.arch == 'gcn':
        graph_dict = {'strategy': 'spatial'}
        encoder = Encoder(in_channels=opt.channels+opt.n_classes,n_z = opt.latent_dim, graph_args=graph_dict)
        decoder = Decoder(in_channels=opt.channels,n_z = opt.latent_dim+opt.n_classes,graph_args=graph_dict)
        encoder.apply(init_weights)
        decoder.apply(init_weights)
    elif opt.arch == 'gcn_org':
        encoder = Encoder_org(channel=opt.channels+opt.n_classes,window_size=opt.img_width,num_point=opt.img_height,num_person=opt.num_per,use_data_bn=False)
        #encoder = Encoder_org(channel=opt.channels,window_size=opt.img_width,num_point=opt.img_height,num_person=opt.num_per)
        decoder = Decoder_org(num_point=opt.img_height,num_person=opt.num_per,n_z=opt.latent_dim+opt.n_classes,use_data_bn=False)
        encoder.apply(init_weights)
        decoder.apply(init_weights)
    elif opt.arch == 'tcn':
        encoder = Encoder_tcn(channel=opt.channels,num_class=1,num_per=opt.num_per,num_joints=opt.img_height,use_data_bn=True)
        decoder = Decoder_tcn(channel=opt.n_classes+opt.latent_dim,num_class=1,num_per=opt.num_per,num_joints=opt.img_height,use_data_bn=False)

    if opt.arch == 'gcn' or opt.arch=='gcn_org' or opt.arch == 'tcn':
        encoder.cuda()
        decoder.cuda()
        adversarial_loss.cuda()
    else:
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
    else:
        traindata = NTURGBDData_full(0,None,0,opt.dataset,'sub',0,0,0,opt.real_per,opt.num_per,'train',opt.expt)
    dataloader = DataLoader(traindata,batch_size=opt.batch_size,shuffle=True,num_workers=opt.n_cpu,pin_memory=True)

    # Optimizers
    if opt.arch == 'gcn' or opt.arch == 'gcn_org' or opt.arch == 'tcn':
        optimizer_G = torch.optim.Adam(decoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        scheduler_G = ExponentialLR(optimizer_G, gamma=0.99, last_epoch=-1)
        scheduler_D = ExponentialLR(optimizer_D, gamma=0.99, last_epoch=-1)
    else:
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        #optimizer_G = torch.optim.SGD(generator.parameters(), lr=opt.lr)
        #optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=opt.lr)
        scheduler_G = ExponentialLR(optimizer_G, gamma=0.99, last_epoch=-1)
        scheduler_D = ExponentialLR(optimizer_D, gamma=0.99, last_epoch=-1)
    # scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=30)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    writer = SummaryWriter(ckpt_dir)
    
    #print ("Arch",opt.arch)

    for epoch in range(opt.n_epochs):
        generator_loss = AverageMeter()
        discriminator_loss = AverageMeter()

        #generator.train()
        #discriminator.train()

        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]

            imgs = imgs[:,:,:,:,0:1]
            #print ("Images",imgs.shape)
            imgs = np.squeeze(imgs)
            #print ("Images",imgs.shape)
            if batch_size == 1:
                imgs = torch.unsqueeze(imgs,0)

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
                valid = Variable(FloatTensor(batch_size).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size).fill_(0.0), requires_grad=False)
            if opt.arch == 'mlp':
                labels = Variable(labels.type(LongTensor))

            if opt.arch == 'cnn':
                real_y = torch.zeros(batch_size, opt.n_classes)
                real_y = real_y.scatter_(1, labels.view(batch_size, 1), 1).view(batch_size, opt.n_classes, 1, 1).contiguous()
                real_y = Variable(real_y.expand(-1, -1, opt.img_width, opt.img_height).cuda())

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            #print ("Arch",opt.arch)

            # Sample noise and labels as generator input
            #print ("Batch size",batch_size)
            if opt.arch == 'mlp':
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
                #gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
                gen_labels = labels
            elif opt.arch == 'cnn':
                noise = Variable(torch.randn((batch_size, opt.latent_dim,1,1)).cuda())
                gen_labels = (torch.rand(batch_size, 1) * opt.n_classes).type(torch.LongTensor)
                #gen_labels = labels
                gen_y = torch.zeros(batch_size, opt.n_classes)
                gen_y = Variable(gen_y.scatter_(1, gen_labels.view(batch_size, 1), 1).view(batch_size, opt.n_classes,1,1).cuda())
                gen_y_for_D = gen_y.view(batch_size, opt.n_classes, 1, 1).contiguous().expand(-1, -1, opt.img_width, opt.img_height)
            elif opt.arch == 'gcn' or opt.arch == 'gcn_org':
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
                #z = to_var(torch.randn([batch_size, opt.latent_dim]))
                labels = labels.cuda()
                y_onehot = torch.FloatTensor(batch_size, opt.n_classes).cuda()
                y_onehot.zero_()
                y_onehot.scatter_(1, labels.view(batch_size,1), 1)
                labels_dec = y_onehot
                real_imgs = torch.unsqueeze(real_imgs,dim=4)
                y_onehot = y_onehot.unsqueeze(2).expand(-1, -1, real_imgs.shape[2])
                y_onehot = y_onehot.unsqueeze(3).expand(-1,-1,-1,real_imgs.shape[3])
                y_onehot = y_onehot.unsqueeze(4).expand(-1, -1, -1, -1, real_imgs.shape[4])
                labels_enc = y_onehot
            elif opt.arch == 'tcn':
                #print ("In TCN",opt.arch)
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
                labels = labels.cuda()
                y_onehot = torch.FloatTensor(batch_size, opt.n_classes).cuda()
                y_onehot.zero_()
                y_onehot.scatter_(1, labels.view(batch_size,1), 1)
                labels_dec = y_onehot
                real_imgs = torch.unsqueeze(real_imgs,dim=4)
                #print (labels[0])
                #print (labels_dec.shape)
                #print (labels_enc.shape)
                #print (labels_dec[0])
                #print (labels_enc[0])
            #print (z)
                
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
            elif opt.arch == 'gcn':
                gen_imgs = decoder(z,labels_dec,opt.img_width,opt.img_height) #z,l,T,V
            elif opt.arch == 'gcn_org':
                gen_imgs = decoder(z,labels_dec,opt.channels,opt.img_width,opt.img_height,opt.num_per)
            elif opt.arch == 'tcn':
                #print ("In TCN",opt.arch)
                gen_imgs = decoder(z,labels_dec,C=opt.channels,T=opt.img_width,V=opt.img_height,M=opt.num_per)
            #print ("Gen imgs",gen_imgs.requires_grad)

            #print ("Arch",opt.arch)

            # Loss measures generator's ability to fool the discriminator
            #discriminator.eval()
            if opt.arch == 'mlp':
                validity = discriminator(gen_imgs, gen_labels)
                #validity = discriminator(gen_imgs)
                g_loss = adversarial_loss(validity, valid)
            elif opt.arch == 'cnn':
                g_loss = adversarial_loss(discriminator(gen_imgs,gen_y_for_D).squeeze(), valid)
            elif opt.arch == 'gcn' or opt.arch == 'gcn_org':
                #print ("In here",opt.arch)
                validity = encoder(gen_imgs,labels_enc)
                #validity = encoder(gen_imgs)
                g_loss = adversarial_loss(validity,valid)
            elif opt.arch == 'tcn':
                validity = encoder(gen_imgs)
                g_loss = adversarial_loss(validity,valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            #discriminator.train()
            #generator.eval()

            optimizer_D.zero_grad()

            # Loss for real images
            if opt.arch == 'mlp':
                validity_real = discriminator(real_imgs, labels)
                #validity_real = discriminator(real_imgs)
                d_real_loss = adversarial_loss(validity_real, valid)
            elif opt.arch == 'cnn':
                #gen_imgs = generator(noise, gen_y)
                d_real_loss = adversarial_loss(discriminator(real_imgs, real_y).squeeze(), valid)
            elif opt.arch == 'gcn' or opt.arch == 'gcn_org':
                validity_real = encoder(real_imgs,labels_enc)
                #validity_real = encoder(real_imgs)
                d_real_loss = adversarial_loss(validity_real,valid)
            elif opt.arch == 'tcn':
                validity_real = encoder(real_imgs)
                d_real_loss = adversarial_loss(validity_real,valid)

            # Loss for fake images
            if opt.arch == 'mlp':
                validity_fake = discriminator(gen_imgs.detach(), gen_labels)
                #validity_fake = discriminator(gen_imgs.detach())
                d_fake_loss = adversarial_loss(validity_fake, fake)
            elif opt.arch == 'cnn':
                d_fake_loss = adversarial_loss(discriminator(gen_imgs.detach(),gen_y_for_D).squeeze(), fake)
            elif opt.arch == 'gcn' or opt.arch == 'gcn_org':
                validity_fake = encoder(gen_imgs.detach(), labels_enc)
                #validity_fake = encoder(gen_imgs.detach())
                d_fake_loss = adversarial_loss(validity_fake,fake)
            elif opt.arch == 'tcn':
                validity_fake = encoder(gen_imgs.detach())
                d_fake_loss = adversarial_loss(validity_fake,fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [LR: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), optimizer_D.param_groups[0]['lr'])
            )

            batches_done = epoch * len(dataloader) + i

            if batches_done % opt.sample_interval == 0:
                """Saves a grid of generated digits ranging from 0 to n_classes"""
                if opt.arch == 'mlp':
                    # Sample noise
                    z = Variable(FloatTensor(np.random.normal(0, 1, (5 ** 2, opt.latent_dim))))
                    # Get labels ranging from 0 to n_classes for n rows
                    labels = np.array([num for _ in range(5) for num in range(5)])
                    labels = Variable(LongTensor(labels))
                    gen_imgs = generator(z, labels)
                    save_image(gen_imgs.data[:25], "%s/%d.png" % (images_dir,batches_done), nrow=3, normalize=False)
                    save_image(gen_imgs.data[:25], "%s/%d_norm.png" % (images_dir,batches_done), nrow=3, normalize=True)
                elif opt.arch == 'cnn':
                    noise = Variable(torch.randn((5, opt.latent_dim,1,1)).cuda())
                    gen_labels = (torch.rand(5, 1) * opt.n_classes).type(torch.LongTensor)
                    gen_y = torch.zeros(5, opt.n_classes)
                    gen_y = Variable(gen_y.scatter_(1, gen_labels.view(5, 1), 1).view(5, opt.n_classes,1,1).cuda())
                    gen_imgs = generator(noise, gen_y)
                    gen_imgs = gen_imgs.permute(0,1,3,2)
                    save_image(gen_imgs.data[:25], "%s/%d.png" % (images_dir,batches_done), nrow=3, normalize=False)
                    save_image(gen_imgs.data[:25], "%s/%d_norm.png" % (images_dir,batches_done), nrow=3, normalize=True)
                elif opt.arch == 'gcn':
                    z = Variable(FloatTensor(np.random.normal(0, 1, (5, opt.latent_dim))))
                    #z = to_var(torch.randn([5, opt.latent_dim]))
                    labels = np.array([num for num in range(5)])
                    labels = Variable(LongTensor(labels))
                    y_onehot = torch.FloatTensor(5, opt.n_classes).cuda()
                    y_onehot.zero_()
                    y_onehot.scatter_(1, labels.view(5,1), 1)
                    labels_dec = y_onehot
                    gen_imgs = decoder(z,labels_dec,opt.img_width,opt.img_height) #z,l,T,V
                    gen_imgs = gen_imgs.squeeze(dim=4)
                elif opt.arch == 'gcn_org':
                    z = Variable(FloatTensor(np.random.normal(0, 1, (5, opt.latent_dim))))
                    #z = to_var(torch.randn([5, opt.latent_dim]))
                    labels = np.array([num for num in range(5)])
                    labels = Variable(LongTensor(labels))
                    y_onehot = torch.FloatTensor(5, opt.n_classes).cuda()
                    y_onehot.zero_()
                    y_onehot.scatter_(1, labels.view(5,1), 1)
                    labels_dec = y_onehot
                    gen_imgs = decoder(z,labels_dec,opt.channels,opt.img_width,opt.img_height,opt.num_per) #z,l,C,T,V,M
                    gen_imgs = gen_imgs.squeeze(dim=4)
                elif opt.arch == 'tcn':
                    z = Variable(FloatTensor(np.random.normal(0, 1, (5, opt.latent_dim))))
                    #z = to_var(torch.randn([5, opt.latent_dim]))
                    labels = np.array([num for num in range(5)])
                    labels = Variable(LongTensor(labels))
                    y_onehot = torch.FloatTensor(5, opt.n_classes).cuda()
                    y_onehot.zero_()
                    y_onehot.scatter_(1, labels.view(5,1), 1)
                    labels_dec = y_onehot
                    gen_imgs = decoder(z,labels_dec,C=opt.channels,T=opt.img_width,V=opt.img_height,M=opt.num_per)
                    gen_imgs = gen_imgs.squeeze(dim=4)
                #print (gen_imgs.shape)
                
                create_video(gen_imgs,batches_done,5,videos_dir)

            if opt.arch == 'gcn' or opt.arch == 'gcn_org' or opt.arch == 'tcn':
                state={
                    'last_epoch': epoch+1,
                    'last_state_dict_G': decoder.state_dict(),
                    'optimizer_G' : optimizer_G.state_dict(),
                    'last_state_dict_D': encoder.state_dict(),
                    'optimizer_D':optimizer_D.state_dict()
                    }
            else:
                state={
                    'last_epoch': epoch+1,
                    'last_state_dict_G': generator.state_dict(),
                    'optimizer_G' : optimizer_G.state_dict(),
                    'last_state_dict_D':discriminator.state_dict(),
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
        
        scheduler_D.step()
        scheduler_G.step()

        # Train log - Summary Writer
        writer.add_scalar('Generator/Loss',generator_loss.avg,epoch)
        writer.add_scalar('Discriminator/Loss', discriminator_loss.avg, epoch)
        writer.add_scalar('Generator/lrate',lrate_G,epoch)
        writer.add_scalar('Discriminator/lrate',lrate_D,epoch)
    
    writer.close()

if opt.mode == 'test':

    init_seed(0)

    # Initialize generator and discriminator
    if opt.arch == 'mlp':
        generator = Generator(latent_dim = opt.latent_dim, img_shape=img_shape, n_classes=opt.n_classes)
        discriminator = Discriminator(img_shape = img_shape, n_classes=opt.n_classes)
    elif opt.arch == 'cnn':
        generator = Generator_cnn(latent_dim=opt.latent_dim,DIM=256,classes=opt.n_classes)
        discriminator = Discriminator_cnn(d=256) 

    if cuda:
        generator.cuda()
        discriminator.cuda()

    ## Model path
    #/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_center_nonorm_per1_ep5k/models/5000/model_state_100.pth
    #/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_per1_ep500/models/100/model_state_100.pth
    #/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_bceloss_sigmoid_per1_ep5k/models/5000/model_state_100.pth
    #/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_changelabels_per1_ep5k/models/5000/model_state_100.pth
    #/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_largemodel_changelabels_per1_5p5k/models/5000/model_state_100.pth
    #/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_mlpsixlayer_changelabels_per1_p5k/models/5000/model_state_100.pth
    #/fast-stripe/workspaces/deval/synthetic-data/cgan/realper10_z512_hcndata_cnn_sigmoid_mseloss_linearblock_per1_ep500/models/100/model_state_10.pth
    #/fast-stripe/workspaces/deval/synthetic-data/cgan/realper10_z512_center_mlp_mseloss_per1_ep5k/models/5000/model_state_10.pth

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
    state_dict_disc = dict['last_state_dict_D']
    discriminator.load_state_dict(state_dict_disc,strict=True)
    discriminator.eval()

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
        
        if opt.arch == 'mlp':
            gen_imgs = generator(z, gen_labels)
        elif opt.arch == 'cnn':
            noise = Variable(torch.randn((opt.eval_samples, opt.latent_dim,1,1)).cuda())
            gen_labels = (torch.rand(opt.eval_samples, 1) * opt.n_classes).type(torch.LongTensor)
            gen_y = torch.zeros(opt.eval_samples, opt.n_classes)
            gen_y = Variable(gen_y.scatter_(1, gen_labels.view(opt.eval_samples, 1), 1).view(opt.eval_samples, opt.n_classes,1,1).cuda())
            gen_imgs = generator(noise, gen_y)
            gen_imgs = gen_imgs.permute(0,1,3,2)
        #validity = discriminator(gen_imgs, gen_labels)
        #print ("Validity",validity)
        #probs_validity = F.sigmoid(validity)
        #print ("Discriminator probs",probs_validity)
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