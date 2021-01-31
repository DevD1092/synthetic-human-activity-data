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
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--expt", type=str, default='check', help="Name of the expt")
parser.add_argument("--temporal_length", type=int, default=60, help="Temporal length")
parser.add_argument("--temporal_pattern", type=str, default=None, help="Temporal pattern")
parser.add_argument("--dataset", type=str, default='hcn', help="hcn,vacnn,dgnn")
parser.add_argument("--normalize", type=int,default=1,help="Normalize the data or not")
parser.add_argument("--save_steps",type=int,default=250,help="Steps interval for model saving")
parser.add_argument("--mode",type=str,default='train',help="train or test mode")
parser.add_argument("--model_path",type=str,default=None,help="Model path for testing mode")
parser.add_argument("--eval_samples",type=int,default=5,help="Number of eval samples to be produced")
opt = parser.parse_args()
print(opt)

ckpt_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/dcgan',opt.expt,'checkpoints')
images_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/dcgan',opt.expt,'images')
video_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/dcgan',opt.expt,'videos')
models_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp',opt.expt,'models')
os.makedirs(images_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

init_seed(0)

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def denormalize_data(data,max_val,min_val,range_num):
    if range_num == 0:
        data = data*(max_val-min_val) + min_val
    else:
        data = (data + 1)*(max_val-min_val)/2 + min_val
    return data
    
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
        vis_3dske(data_3d,opt.temporal_length,vid_save_dir,i,opt.normalize)

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

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        if opt.temporal_pattern == 'interpolate':
            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
                nn.ReplicationPad2d((1,0,1,0)),
                nn.Tanh(),
            )
        else:
            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
                nn.Tanh(),
            )

    def forward(self, z):
        #print ("Init size shape",self.init_size)
        #print ("Z",z.shape)
        out = self.l1(z)
        #print ("L1 out",out.shape)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        #print ("Out reshaped",out.shape)
        img = self.conv_blocks(out)
        #print ("Final image",img.shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        self.ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * self.ds_size ** 2, 1), nn.Sigmoid())
        if (opt.temporal_pattern == 'interpolate'):
            #self.adv_layer = nn.Sequential(nn.Linear(128 * 4, 1), nn.Sigmoid())
            self.adv_layer = nn.Sequential(nn.Linear(128 * 4, 1)) # Do not use Sigmoid

    def forward(self, img):
        #print ("Down sampled img size",self.ds_size)
        #print ("Image",img.shape)
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        #print ("Out ",out.shape)
        validity = self.adv_layer(out)
        #print ("Validity",validity.shape)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()
#adversarial_loss = torch.nn.BCEWithLogitsLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# ----------
#  Training
# ----------

if opt.mode == 'train':

    # Configure data loader
    if opt.expt == 'check':
        opt.n_cpu = 0
        opt.n_epochs = 1
    if opt.temporal_pattern == 'interpolate':
        traindata = NTURGBDData(opt.temporal_length,opt.temporal_pattern,0,opt.dataset,'sub',opt.normalize,'train',opt.expt)
    else:
        traindata = NTURGBDData(0,None,0,opt.dataset,'sub','train',opt.expt)
    dataloader = DataLoader(traindata,batch_size=opt.batch_size,shuffle=True,num_workers=opt.n_cpu,pin_memory=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    writer = SummaryWriter(ckpt_dir)

    for epoch in range(opt.n_epochs):
        generator_loss = AverageMeter()
        discriminator_loss = AverageMeter()
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
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
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i

            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "%s/%d.png" % (images_dir,batches_done), nrow=5, normalize=False)
                save_image(gen_imgs.data[:25], "%s/%d_norm.png" % (images_dir,batches_done), nrow=5, normalize=True)
                create_video(gen_imgs,batches_done,5,video_dir)

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
                model_write_path = os.path.join(model_write_path,'model_state.pth')
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

# ----------
#  Testing
# ----------

if opt.mode == 'test':
    dict = torch.load(opt.model_path)
    state_dict = dict['last_state_dict_G']
    generator.load_state_dict(state_dict,strict=True)
    generator.eval()

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    videos_dir = opt.model_path.replace('model_state.pth','eval_op')
    os.makedirs(videos_dir,exist_ok=True)

    for i in range(opt.eval_samples):
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
        fake_imgs = generator(z)
        #print (fake_imgs.shape)
        create_video(fake_imgs,i,1,videos_dir)
        print ("Output written for sample",i)
    print ("Evaluation completed")
