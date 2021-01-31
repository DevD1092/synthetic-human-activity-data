import argparse
import os
import numpy as np
import math
import sys
import random
sys.path.append('../')
sys.path.append('../hcn/')
import subprocess

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from tensorboardX import SummaryWriter
from dataloader import NTURGBDData,vis_3dske
from model import Generator,Discriminator,Generator1,Discriminator1,Generator_concat512
from HCN_model import HCN4,HCN,HCN2,HCN3,HCN1per_feat,HCN1per_feat_1 #HCN4 - single person
from losses import ContrastiveLoss, TripletLoss
from tqdm import tqdm

## Command
# Train -> python wgan_gp_action_class.py --img_width 60 --img_height 25 --channels 3 --temporal_length 60 --temporal_pattern interpolate --num_per 1 --centering 1 --sample_interval 5000 --save_steps 500 --expt img25_lambdatriplet_1p0_per1_ep5k --n_epochs 5000 --class_batch 0 --feat_loss triplet --margin 2.0 --feat_concat 0
# Test -> python wgan_gp_action_class.py --mode test --temporal_length 60 --temporal_pattern interpolate --img_width 60 --img_height 25 --channels 3 --class_batch 0 --eval_samples 2048

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
parser.add_argument("--n_classes", type=int, default=49, help="number of classes for dataset")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_width", type=int, default=25, help="image width dimension")
parser.add_argument("--img_height", type=int, default=25, help="image height dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
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
parser.add_argument("--dataset",type=str,default='hcn',help="Which ntu dataset - hcn or vacnn")
parser.add_argument("--feat_loss",type=str,default='contr',help="Which type of feature loss - contrastive or triplet")
parser.add_argument("--update_feat",type=int,default=0,help="Update the feat network or not")
parser.add_argument("--feat_concat",type=int,default=0,help="0 - No concat; 256- concat 256&512; 512 - concat 512 only; -1 - concat 512 early")
parser.add_argument("--lambda_contr",type=float,default=1.0,help="Contrastive loss hyperparameter")
parser.add_argument("--margin",type=float,default=2.0,help="Margin value for triplet and cosine loss")
opt = parser.parse_args()
print(opt)

if opt.class_batch != -1:
    start_ind = opt.class_batch*4
    end_ind = (opt.class_batch+1)*4
    classes = [i for i in range(start_ind,end_ind)]
    if opt.class_batch == 11:
        classes.append(48)
    #elif opt.class_batch == 1:
    #    classes = [5, 4]
    #print (classes)
else:
    classes = [42]

img_shape = (opt.channels, opt.img_width, opt.img_height)

ntu_singleper_classdist = {25: 672, 37: 672, 38: 672, 24: 672, 46: 672, 45: 672, 26: 672, 40: 672, 41: 671, 35: 671, 31: 671, 42: 671, 33: 671, 34: 671, 47: 671, 48: 671, 44: 671, 43: 670, 23: 670, 32: 670, 36: 670, 22: 670, 27: 669, 14: 669, 6: 669, 21: 669, 29: 669, 16: 669, 20: 669, 28: 669, 7: 668, 9: 668, 17: 668, 30: 668, 13: 668, 18: 668, 5: 668, 4: 667, 15: 667, 3: 667, 12: 667, 19: 667, 0: 666, 1: 666, 2: 665, 10: 664, 8: 663, 11: 660, 39: 660}

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

def get_mean_std_0(loader_):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for i,(data,_) in enumerate(loader_):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print("Mean",mean,"Std",std)

def get_mean_std(loader_):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader_):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    #print ("Mean",mean,"Std",std)
    return mean,std

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ----------
#  Training
# ----------

if opt.mode == 'train':

    init_seed(0)

    for class_num in classes:

        ckpt_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp_action_class',str(class_num),opt.expt,'checkpoints')
        images_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp_action_class',str(class_num),opt.expt,'images')
        videos_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp_action_class',str(class_num),opt.expt,'videos')
        models_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp_action_class',str(class_num),opt.expt,'models')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(videos_dir,exist_ok=True)

        #HCN Pre-trained model
        if opt.temporal_length == 25:
            fc7_dim = 2048
        elif opt.temporal_length == 60:
            fc7_dim = 4096
        if opt.num_per == 1:
            hcn_model = HCN1per_feat_1(in_channel=3,num_person=1,num_class=49,fc7_dim=fc7_dim).cuda()
        elif opt.num_per == 2:
            hcn_model = HCN(in_channel=3,num_person=2,num_class=49,fc7_dim=fc7_dim).cuda() 
        hcn_model_path = '/fast-stripe/workspaces/deval/synthetic-data/hcn/CCC_models/clall_len60_per1/model_state.pth'
        hcn_model.load_state_dict(torch.load(hcn_model_path)['best_test_model'], strict=False)
        print ("Loaded model",hcn_model_path,"| Best test per",torch.load(hcn_model_path)['best_test_perf'],"| Best test epoch",torch.load(hcn_model_path)['best_test_epoch'])

        # Loss weight for gradient penalty
        lambda_gp = opt.lambda_gp

        # Initialize generator and discriminator
        if opt.arch == 'mlp':
            if opt.feat_concat == 0:
                generator = Generator(latent_dim=opt.latent_dim,img_shape=img_shape)
            elif opt.feat_concat == 512:
                generator = Generator_concat512(latent_dim=opt.latent_dim,img_shape=img_shape)
            discriminator = Discriminator(img_shape=img_shape)
        elif opt.arch == 'cnn':
            generator = Generator1(latent_dim=opt.latent_dim)
            discriminator = Discriminator1()

        triplet_loss_torch = torch.nn.TripletMarginLoss(margin=opt.margin) #2.0 default
        cosine_embedding_loss = torch.nn.CosineEmbeddingLoss(margin=opt.margin) #0.5 default

        if cuda:
            generator.cuda()
            discriminator.cuda()
            triplet_loss_torch.cuda()
            cosine_embedding_loss.cuda()
        
        # Configure data loader
        transform = None
        if opt.centering == 0 and opt.num_per == 1:
            mean = [ 0.0052, -0.4731,  3.1408]
            std = [0.4781, 0.4572, 0.5331]
        elif opt.centering == 1 and opt.num_per == 1:
            mean = [-0.0049, -0.2049, -0.0329]
            std = [0.1714, 0.3030, 0.1716]
        elif opt.centering == 0 and opt.num_per == 2:
            mean = [ 0.0068, -0.2482,  1.6656]
            std = [0.3621, 0.4055, 1.6215]
        elif opt.centering == 1 and opt.num_per == 2:
            mean = [-0.0073,  0.0204, -1.5064]
            std = [0.3686, 0.4308, 1.6063]
        if opt.transform == 1:
            #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean,std=std)])
            transform = transforms.Compose([transforms.Normalize(mean=mean,std=std)])
        if opt.expt == 'check':
            opt.n_cpu = 0
        if opt.temporal_pattern == 'interpolate':
            traindata = NTURGBDData(opt.temporal_length,'interpolate',0,opt.dataset,'sub',opt.normalize,opt.centering,opt.spherical,class_num,opt.real_per,opt.num_per,'train',transform,opt.expt)
        else:
            traindata = NTURGBDData(0,None,0,opt.dataset,'sub',0,0,0,class_num,opt.real_per,opt.num_per,'train',opt.expt)
        dataloader = DataLoader(traindata,batch_size=opt.batch_size,shuffle=True,num_workers=opt.n_cpu,pin_memory=True)

        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        writer = SummaryWriter(ckpt_dir)

        batches_done = 0
        for epoch in range(opt.n_epochs):
            generator_loss = AverageMeter()
            discriminator_loss = AverageMeter()
            for i, (imgs, _) in enumerate(dataloader):

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))
                batch_size = imgs.shape[0]

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

                # Generate a batch of images
                if opt.feat_concat == 0:
                    fake_imgs = generator(z)
                elif opt.feat_concat == 512:
                    out1 = np.zeros((batch_size,512),dtype=np.float32)
                    for ind in range(batch_size):
                        feat_maps_path = '/fast-stripe/workspaces/deval/synthetic-data/hcn/feat_map/'
                        feat_maps_path = os.path.join(feat_maps_path,str(class_num),'feat_map.npy')
                        feat_maps = np.load(feat_maps_path)
                        out1[ind] = feat_maps
                    out1 = torch.from_numpy(out1).cuda()
                    fake_imgs = generator(z, out1)

                # Real images
                real_validity = discriminator(real_imgs)
                # Fake images
                fake_validity = discriminator(fake_imgs)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

                d_loss.backward()
                optimizer_D.step()

                optimizer_G.zero_grad()

                # Train the generator every n_critic steps
                if i % opt.n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                    if opt.feat_concat == 0:
                        fake_imgs = generator(z)
                    elif opt.feat_concat == 512:
                        out1 = np.zeros((batch_size,512),dtype=np.float32)
                        for ind in range(batch_size):
                            feat_maps_path = '/fast-stripe/workspaces/deval/synthetic-data/hcn/feat_map/'
                            feat_maps_path = os.path.join(feat_maps_path,str(class_num),'feat_map.npy')
                            feat_maps = np.load(feat_maps_path)
                            out1[ind] = feat_maps
                        out1 = torch.from_numpy(out1).cuda()
                        fake_imgs = generator(z, out1)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = discriminator(fake_imgs)

                    if opt.feat_concat == 0:
                    ## With pre-saved feature maps
                        out1 = np.zeros((batch_size,512),dtype=np.float32) # Positive samples
                        if opt.feat_loss == 'triplet' or opt.feat_loss == 'cosineembed':
                            out3 = np.zeros((batch_size,512),dtype=np.float32) # Negative samples
                        #print (gen_labels_numpy)
                        for ind in range(batch_size):
                            feat_maps_path = '/fast-stripe/workspaces/deval/synthetic-data/hcn/feat_map/'
                            feat_maps_path = os.path.join(feat_maps_path,str(class_num),'feat_map.npy')
                            feat_maps = np.load(feat_maps_path)
                            out1[ind] = feat_maps
                            if opt.feat_loss == 'triplet' or opt.feat_loss == 'cosineembed':
                                neg_labels = [class_lab for class_lab in range(0,opt.n_classes)]
                                neg_labels.remove(class_num)
                                neg_label = random.choice(neg_labels)                    
                                feat_maps_path = '/fast-stripe/workspaces/deval/synthetic-data/hcn/feat_map/'
                                feat_maps_path = os.path.join(feat_maps_path,str(neg_label),'feat_map.npy')
                                feat_maps = np.load(feat_maps_path)
                                out3[ind] = feat_maps

                        out1 = torch.from_numpy(out1).cuda()
                        out3 = torch.from_numpy(out3).cuda()
                        #print (out1)
                        #print (out1.shape)
                        similar = batch_size
                        dissimilar = 0

                        if opt.num_per == 1:
                            #real_imgs = np.expand_dims(real_imgs,axis=3)
                            #gen_imgs = np.expand_dims(gen_imgs,axis=3)
                            sec_per = np.zeros((batch_size,3,opt.temporal_length,25,1),dtype=np.float32)
                            sec_per = torch.from_numpy(sec_per).cuda()
                            #real_imgs_hcn = torch.unsqueeze(real_imgs,4)
                            gen_imgs_hcn = torch.unsqueeze(fake_imgs,4)
                            #real_imgs_hcn = torch.cat((real_imgs_hcn,sec_per),-1)
                            gen_imgs_hcn = torch.cat((gen_imgs_hcn,sec_per),-1)
                            #print (real_imgs.shape)
                            #print (gen_imgs.shape)
                            #print (sec_per.shape)

                        out2 = hcn_model(gen_imgs_hcn)

                        if opt.feat_loss == 'contr':
                            contrastive_loss = ContrastiveLoss(margin=2.0)
                            contrastive_loss = contrastive_loss(out1,out2,intersec_labels)
                            feat_loss = contrastive_loss

                        elif opt.feat_loss == 'triplet':
                            #triplet_loss = TripletLoss(margin=2.0)
                            #triplet_loss = triplet_loss(anchor=out2,positive=out1,negative=out3)
                            triplet_loss = triplet_loss_torch(anchor=out2,positive=out1,negative=out3)
                            #print ("Triplet loss manual",triplet_loss)
                            #print ("Triplet loss torch",triplet_loss_torch)
                            feat_loss = triplet_loss
                        elif opt.feat_loss == 'cosineembed':
                            pos = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                            neg = Variable(FloatTensor(batch_size, 1).fill_(-1.0), requires_grad=False)
                            cosine_loss_pos = cosine_embedding_loss(out2,out1,pos)
                            cosine_loss_neg = cosine_embedding_loss(out2,out3,neg)
                            feat_loss = cosine_loss_pos + cosine_loss_neg   

                    if opt.feat_concat == 0:
                        g_loss = -torch.mean(fake_validity) + opt.lambda_contr * feat_loss
                    elif opt.feat_concat == 512:
                        g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    optimizer_G.step()

                    if opt.feat_concat == 0:

                        if opt.feat_loss == 'triplet':
                            print(
                                "[Class num %d] [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Triplet loss: %f]"
                                % (class_num,epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), feat_loss.item())
                            )
                        
                        elif opt.feat_loss == 'cosineembed':
                            print(
                                "[Class num %d] [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Cosine loss: %f]"
                                % (class_num,epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), feat_loss.item())
                            )
                    
                    elif opt.feat_concat == 512:
                        
                        print(
                                "[Class num %d] [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                                % (class_num,epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                            )

                    if batches_done % opt.sample_interval == 0:
                        save_image(fake_imgs.data[:25], "%s/%d.png" % (images_dir,batches_done), nrow=5, normalize=False)
                        save_image(fake_imgs.data[:25], "%s/%d_norm.png" % (images_dir,batches_done), nrow=5, normalize=True)
                        create_video(fake_imgs,batches_done,5,videos_dir)

                    state={
                        'last_epoch': epoch+1,
                        'last_state_dict_G': generator.state_dict(),
                        'optimizer_G' : optimizer_G.state_dict(),
                        'last_state_dict_D': discriminator.state_dict(),
                        'optimizer_D':optimizer_D.state_dict(),
                        'last_state_dict_hcn': hcn_model.state_dict()
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

                    batches_done += opt.n_critic
                
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
    
    init_seed(0)

    # Initialize generator and discriminator
    if opt.arch == 'mlp':
        generator = Generator(latent_dim=opt.latent_dim,img_shape=img_shape)
        discriminator = Discriminator(img_shape=img_shape)
    elif opt.arch == 'cnn':
        generator = Generator1(latent_dim=opt.latent_dim)
        discriminator = Discriminator1()

    if cuda:
        generator.cuda()
        discriminator.cuda()

    if opt.class_batch == -1:

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
            print ("Video output written for sample",i+1)
        print ("Evaluation completed")

    else:
        start_ind = opt.class_batch*4
        end_ind = (opt.class_batch+1)*4
        classes = [i for i in range(start_ind,end_ind)]
        if opt.class_batch == 11:
            classes.append(48)

        for class_num in classes:
            print ("Class",class_num)
            # Path /fast-stripe/workspaces/deval/synthetic-data/wgan_gp_action_class/0/img25_lambdatriplet_1p0_per1_ep5k/models/5000
            # Path /fast-stripe/workspaces/deval/synthetic-data/wgan_gp_action_class/0/img25_lambdatriplet_5p0_per1_ep5k/models/5000
            model_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp_action_class/',str(class_num))
            model_path = os.path.join(model_path,'img25_lambdatriplet_5p0_per1_ep5k/models/5000/model_state_%d.pth'%opt.real_per)
            dict = torch.load(model_path)
            print ("Loaded the model",model_path)
            state_dict = dict['last_state_dict_G']
            generator.load_state_dict(state_dict,strict=True)
            generator.eval()

            Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

            output_dir = model_path.replace('model_state_%d.pth'%opt.real_per,'eval_op')
            os.makedirs(output_dir,exist_ok=True)

            # Calc the number of samples to be added
            #for syn_per in tqdm([25,50,75,100]):
            #num_real_samples = ntu_singleper_classdist[class_num]
            #num_syn_samples = int((syn_per/100)*num_real_samples)
            num_syn_samples = opt.eval_samples
                #print ("Per",syn_per,"Samples",num_syn_samples)
            syn_data = np.zeros((num_syn_samples,3,opt.temporal_length,25,2),dtype=np.float32)
            #output_file = os.path.join(output_dir,'syn_data_%d.npy'%syn_per)
            output_file = os.path.join(output_dir,'syn_%d_samples.npy'%opt.eval_samples)

            for i in range(num_syn_samples):
                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
                fake_imgs = generator(z)
                #print (fake_imgs.shape)

                data_to_save = fake_imgs.data[0:1]
                data_3d = data_to_save[0].cpu().detach().numpy()
                #print (data_3d.shape)
                if opt.num_per == 2:
                    data_per1 = data_3d[:,:,0:25]
                    data_per2 = data_3d[:,:,25:50]
                    data_per1 = np.expand_dims(data_per1,axis=3)
                    data_per2 = np.expand_dims(data_per2,axis=3)
                    syn_data[i,:,:,:,0:1] = data_per1
                    syn_data[i,:,:,:,1:2] = data_per2
                elif opt.num_per == 1:
                    data_3d = np.expand_dims(data_3d,axis=3)
                    syn_data[i,:,:,:,0:1] = data_3d
            
            np.save(output_file,syn_data)
            
            print ("Completed data saving")