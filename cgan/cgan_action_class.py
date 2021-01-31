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
import torch

from tensorboardX import SummaryWriter
from dataloader import NTURGBDData,vis_3dske,NTURGBDData_full
from model import Generator,Discriminator,Generator_concat, Generator_concat256,Generator_earlyconcat
from HCN_model import HCN4,HCN,HCN2,HCN3,HCN1per_feat,HCN1per_feat_1 #HCN4 - single person
from losses import ContrastiveLoss, TripletLoss
from tqdm import tqdm

## Command
# Train -> python cgan_action_class.py --img_width 60 --img_height 25 --channels 3 --temporal_length 60 --temporal_pattern interpolate --expt img25_center_nonorm_lambdacontr_0p1_featmapsnew_per1_ep5k --centering 1 --n_epochs 5000 --sample_interval 80000 --save_steps 500 --batch_size 64 --num_per 1 --transform 0 --lambda_contr 0.1 --feat_loss triplet --margin 2.0 --feat_concat 1
# Test -> python cgan_action_class.py --mode test --img_width 60 --img_height 25 --channels 3 --temporal_length 60 --temporal_pattern interpolate --num_per 1 --eval_samples 2048 --feat_concat 0 --class_batch 0

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
parser.add_argument("--feat_loss",type=str,default='contr',help="Which type of feature loss - contrastive or triplet")
parser.add_argument("--update_feat",type=int,default=0,help="Update the feat network or not")
parser.add_argument("--feat_concat",type=int,default=0,help="0 - No concat; 256- concat 256&512; 512 - concat 512 only; -1 - concat 512 early")
parser.add_argument("--lambda_contr",type=float,default=0.2,help="Contrastive loss hyperparameter")
parser.add_argument("--margin",type=float,default=2.0,help="Margin value for triplet and cosineembed loss")
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

# ----------
#  Training
# ----------

if opt.mode == 'train':

    init_seed(0)

    ckpt_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class',opt.expt,'checkpoints')
    images_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class',opt.expt,'images')
    videos_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class',opt.expt,'videos')
    models_dir = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class',opt.expt,'models')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(videos_dir,exist_ok=True)
    os.makedirs(models_dir,exist_ok=True)

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

    # Loss functions
    adversarial_loss = torch.nn.MSELoss()
    triplet_loss_torch = torch.nn.TripletMarginLoss(margin=opt.margin) #2.0 default
    cosine_embedding_loss = torch.nn.CosineEmbeddingLoss(margin=opt.margin) #0.5 default
    #adversarial_loss = torch.nn.BCEWithLogitsLoss()
    #print ("WARNING: USING BCE LOSS")

    # Initialize generator and discriminator
    if opt.feat_concat == 0:
        generator = Generator(latent_dim = opt.latent_dim, img_shape=img_shape, n_classes=opt.n_classes)
    elif opt.feat_concat == 512:
        generator = Generator_concat(latent_dim=opt.latent_dim, img_shape=img_shape, n_classes=opt.n_classes)
    elif opt.feat_concat == 256:
        generator = Generator_concat256(latent_dim=opt.latent_dim, img_shape=img_shape, n_classes=opt.n_classes)
    elif opt.feat_concat == -1:
        generator = Generator_earlyconcat(latent_dim=opt.latent_dim, img_shape=img_shape, n_classes=opt.n_classes)
    discriminator = Discriminator(img_shape = img_shape, n_classes=opt.n_classes)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        triplet_loss_torch.cuda()
        cosine_embedding_loss.cuda()

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
        traindata = NTURGBDData_full(opt.temporal_length,'interpolate',0,'hcn','sub',opt.normalize,opt.centering,opt.spherical,opt.real_per,opt.num_per,'train',transform,opt.expt)
    else:
        traindata = NTURGBDData_full(0,None,0,'hcn','sub',0,0,0,opt.real_per,opt.num_per,'train',opt.expt)
    dataloader = DataLoader(traindata,batch_size=opt.batch_size,shuffle=True,num_workers=opt.n_cpu,pin_memory=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    if opt.update_feat == 1:
        torch.autograd.set_detect_anomaly(True)
        optimizer_feat = torch.optim.Adam(hcn_model.parameters(), lr=0.001, betas=(0.9, 0.999),eps=1e-8,weight_decay=1e-4)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    writer = SummaryWriter(ckpt_dir)

    for epoch in range(opt.n_epochs):
        generator_loss = AverageMeter()
        discriminator_loss = AverageMeter()
        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            #print ("Batch size",batch_size)
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            #gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
            gen_labels = labels
            #gen_labels = []
            #for j in range(0,batch_size):
            #    gen_labels.append(j)
            #gen_labels = np.array(gen_labels)
            #gen_labels = torch.from_numpy(gen_labels).cuda()
            #print (gen_labels.cpu().numpy())

            # Intersection labels for Contrastive loss
            #intersec_labels = torch.eq(gen_labels,labels).long()
            #similar = np.count_nonzero(intersec_labels.cpu().numpy())
            #dissimilar = batch_size - similar
            #print (non_zeros)

            # Generate a batch of images
            #print ("Latent dim",z.shape)
            #print ("Generator labels",gen_labels.shape)
            if opt.feat_concat == 512 or opt.feat_concat == 256 or opt.feat_concat == -1:
                gen_labels_numpy = gen_labels.cpu().numpy()
                ind = 0
                out1 = np.zeros((batch_size,512),dtype=np.float32)
                if opt.feat_concat == 256:
                    out2 = np.zeros((batch_size,256),dtype=np.float32)
                for label in gen_labels_numpy:
                    feat_maps_path = '/fast-stripe/workspaces/deval/synthetic-data/hcn/feat_map/'
                    feat_maps_path = os.path.join(feat_maps_path,str(label),'feat_map.npy')
                    feat_maps = np.load(feat_maps_path)
                    out1[ind] = feat_maps
                    if opt.feat_concat == 256:
                        feat_maps_path = '/fast-stripe/workspaces/deval/synthetic-data/hcn/feat_map/'
                        feat_maps_path = os.path.join(feat_maps_path,str(label),'feat256_map.npy')
                        feat_maps = np.load(feat_maps_path)
                        out2[ind] = feat_maps
                    ind = ind + 1
                out1 = torch.from_numpy(out1).cuda()
                if opt.feat_concat == 512 or opt.feat_concat == -1:
                    gen_imgs = generator(z, gen_labels, out1)
                if opt.feat_concat == 256:
                    out2 = torch.from_numpy(out2).cuda()
                    gen_imgs = generator(z, gen_labels, out1, out2)
            else:
                gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            
            #Add the contrastive loss for HCN Siamese
            if opt.feat_concat == 0:
                if opt.num_per == 1:
                    #real_imgs = np.expand_dims(real_imgs,axis=3)
                    #gen_imgs = np.expand_dims(gen_imgs,axis=3)
                    #sec_per = np.zeros((batch_size,3,opt.temporal_length,25,1),dtype=np.float32)
                    #sec_per = torch.from_numpy(sec_per).cuda()
                    #real_imgs_hcn = torch.unsqueeze(real_imgs,4)
                    gen_imgs = torch.unsqueeze(gen_imgs,4)
                    #real_imgs_hcn = torch.cat((real_imgs_hcn,sec_per),-1)
                    #gen_imgs_hcn = torch.cat((gen_imgs_hcn,sec_per),-1)
                    #print (real_imgs.shape)
                    #print (gen_imgs.shape)
                    #print (sec_per.shape)
                
                ## With intersec labels
                #with torch.no_grad():
                #    out1, out2 = hcn_model(real_imgs_hcn,gen_imgs_hcn)
                
                ## With pre-saved feature maps
                intersec_labels = Variable(LongTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                gen_labels_numpy = gen_labels.cpu().numpy()
                out1 = np.zeros((batch_size,512),dtype=np.float32) # Positive samples
                if opt.feat_loss == 'triplet' or opt.feat_loss == 'cosineembed':
                    out3 = np.zeros((batch_size,512),dtype=np.float32) # Negative samples
                #print (gen_labels_numpy)
                ind = 0
                for label in gen_labels_numpy:
                    feat_maps_path = '/fast-stripe/workspaces/deval/synthetic-data/hcn/feat_map/'
                    feat_maps_path = os.path.join(feat_maps_path,str(label),'feat_map.npy')
                    feat_maps = np.load(feat_maps_path)
                    out1[ind] = feat_maps
                    if opt.feat_loss == 'triplet' or opt.feat_loss == 'cosineembed':
                        neg_labels = [class_lab for class_lab in range(0,opt.n_classes)]
                        neg_labels.remove(label)
                        neg_label = random.choice(neg_labels)                    
                        feat_maps_path = '/fast-stripe/workspaces/deval/synthetic-data/hcn/feat_map/'
                        feat_maps_path = os.path.join(feat_maps_path,str(neg_label),'feat_map.npy')
                        feat_maps = np.load(feat_maps_path)
                        out3[ind] = feat_maps
                    ind = ind + 1
                out1 = torch.from_numpy(out1).cuda()
                out3 = torch.from_numpy(out3).cuda()
                #print (out1)
                #print (out1.shape)
                similar = batch_size
                dissimilar = 0

                hcn_model.eval()
                #with torch.no_grad():
                out2 = hcn_model(gen_imgs)
                
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
                g_loss = adversarial_loss(validity, valid) + opt.lambda_contr * feat_loss
            elif opt.feat_concat == 256 or opt.feat_concat == 512 or opt.feat_concat == -1:
                g_loss = adversarial_loss(validity,valid)
            #print (feat_loss.requires_grad)
            #g_loss = opt.lambda_contr * feat_loss

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # ---------------------
            #  Train Feat map network
            # ---------------------

            if opt.update_feat == 1:
                out2 = hcn_model(gen_imgs.detach())
                triplet_loss_hcn = triplet_loss_torch(anchor=out2,positive=out1,negative=out3)
                optimizer_feat.zero_grad()
                triplet_loss_hcn.backward()
                optimizer_feat.step()

            if opt.feat_concat == 0:
                if opt.feat_loss == 'contr':
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Contrastive loss: %f] [Similar: %d] [Dissimilar: %d]"
                        % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), feat_loss, similar, dissimilar)
                    )
                elif opt.feat_loss == 'triplet':
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Triplet loss: %f] [Similar: %d] [Dissimilar: %d]"
                        % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), feat_loss, similar, dissimilar)
                    )
                elif opt.feat_loss == 'cosineembed':
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Cosine Embed loss: %f] [Similar: %d] [Dissimilar: %d]"
                        % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), feat_loss, similar, dissimilar)
                    )
            elif opt.feat_concat == 256 or opt.feat_concat == 512 or opt.feat_concat == -1:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                    )

            batches_done = epoch * len(dataloader) + i

            if batches_done % opt.sample_interval == 0:
                """Saves a grid of generated digits ranging from 0 to n_classes"""
                # Sample noise
                z = Variable(FloatTensor(np.random.normal(0, 1, (5, opt.latent_dim))))
                # Get labels ranging from 0 to n_classes for n rows
                #labels = np.array([num for _ in range(5) for num in range(5)])
                labels = np.array([num for num in range(5)])
                labels = Variable(LongTensor(labels))
                if opt.feat_concat == 0:
                    gen_imgs = generator(z, labels)
                elif opt.feat_concat == 256 or opt.feat_concat == 512 or opt.feat_concat == -1:
                    gen_labels = labels
                    gen_labels_numpy = labels.cpu().numpy()
                    ind = 0
                    out1 = np.zeros((5,512),dtype=np.float32)
                    if opt.feat_concat == 256:
                        out2 = np.zeros((5,256),dtype=np.float32)
                    for label in gen_labels_numpy:
                        feat_maps_path = '/fast-stripe/workspaces/deval/synthetic-data/hcn/feat_map/'
                        feat_maps_path = os.path.join(feat_maps_path,str(label),'feat_map.npy')
                        feat_maps = np.load(feat_maps_path)
                        out1[ind] = feat_maps
                        if opt.feat_concat == 256:
                            feat_maps_path = '/fast-stripe/workspaces/deval/synthetic-data/hcn/feat_map/'
                            feat_maps_path = os.path.join(feat_maps_path,str(label),'feat256_map.npy')
                            feat_maps = np.load(feat_maps_path)
                            out2[ind] = feat_maps
                        ind = ind + 1
                    out1 = torch.from_numpy(out1).cuda()
                    if opt.feat_concat == 512 or opt.feat_concat == -1:
                        gen_imgs = generator(z, gen_labels, out1)
                    elif opt.feat_concat == 256:
                        out2 = torch.from_numpy(out2).cuda()
                        gen_imgs = generator(z, gen_labels, out1, out2)
                save_image(gen_imgs.data[:25], "%s/%d.png" % (images_dir,batches_done), nrow=3, normalize=False)
                save_image(gen_imgs.data[:25], "%s/%d_norm.png" % (images_dir,batches_done), nrow=3, normalize=True)
                create_video(gen_imgs,batches_done,5,videos_dir)
            
            if opt.update_feat == 1:
                state={
                    'last_epoch': epoch+1,
                    'last_state_dict_G': generator.state_dict(),
                    'optimizer_G' : optimizer_G.state_dict(),
                    'last_state_dict_D': discriminator.state_dict(),
                    'optimizer_D':optimizer_D.state_dict(),
                    'last_state_dict_hcn':hcn_model.state_dict(),
                    'optimizer_hcn': optimizer_feat.state_dict()
                    }
            else:
                state={
                    'last_epoch': epoch+1,
                    'last_state_dict_G': generator.state_dict(),
                    'optimizer_G' : optimizer_G.state_dict(),
                    'last_state_dict_D': discriminator.state_dict(),
                    'optimizer_D':optimizer_D.state_dict(),
                    'last_state_dict_hcn':hcn_model.state_dict()
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

if opt.mode == 'test':

    init_seed(0)

    # Initialize generator and discriminator
    if opt.feat_concat == 0:
        generator = Generator(latent_dim = opt.latent_dim, img_shape=img_shape, n_classes=opt.n_classes)
    elif opt.feat_concat == 512:
        generator = Generator_concat(latent_dim=opt.latent_dim, img_shape=img_shape, n_classes=opt.n_classes)
    elif opt.feat_concat == -1:
        generator = Generator_earlyconcat(latent_dim=opt.latent_dim, img_shape=img_shape, n_classes=opt.n_classes)
    elif opt.feat_concat == 256:
        generator = Generator_concat256(latent_dim=opt.latent_dim, img_shape=img_shape, n_classes=opt.n_classes)

    if cuda:
        generator.cuda()

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
            print ("Class num: ",class_num)
            # Path 1 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_lambdacontr_0p1_featmapsnew_per1_ep5k/models/5000
            # Path 2 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_lambdatriplet_0p1_featmapsnew_per1_ep5k/models/5000/
            # Path 3 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_cosineembed_0p5_featmapsnew_per1_ep5k/models/2000
            # Path 4 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_cosineembed_0p5_featmapsnew_per1_ep5k/models/1000
            # Path 5 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_torchtriplet_5p0_featmapsnew_per1_ep5k/models/4000
            # Path 6 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_torchtriplet_1p0_featmapsnew_per1_ep5k/models/4000
            # Path 7 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_torchtriplet_0p1_featmapsnew_per1_ep5k/models/4000
            # Path 8 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_lambdacontr1p0_requiregrad_per1_ep5k/models/500
            # Path 9 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_lambdatrip1p0_detach_per1_ep5k/models/500
            # Path 10 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_lambdatrip1p0_updatefeatnw_per1_ep5k/models/500
            # Path 11 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_lambdatrip1p0_updatefeatnw_per1_ep5k/models/2500
            # Path 12 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/CCC_models/cgan_action_class/img25_lambdatrip5p0_margin2p0/models/2500
            # Path 13 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_lambdatrip1p0_updatefeatnw_per1_ep5k/models/4000
            # Path 14 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/CCC_models/cgan_action_class/img25_lambdatriporg1p0_margin2p0/models/2500
            # Path 15 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_lambdacosine1p0_margin2p0_per1_ep5k/models
            # Path 16 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_hcnmodeleval_lambdatriplet_1p0_per1_ep5k/models/2000
            # Path 17 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_concatfeatmap_per1_ep5k/models/5000
            # Path 18 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_hcnmodeleval_lambdatriplet_1p0_margin5p0_per1_ep5k/models/500
            # Path 19 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_concatfeatmap_bcewithlogits_per1_ep5k/models/500
            # Path 20 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_earlyconcat512_per1_ep5k/models/5000
            # Path 21 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_concatfeatmap256_per1_ep5k/models/5000
            # Path 22 /fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_earlyconcatadd512_per1_ep5k/models/5000
            model_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_earlyconcatadd512_per1_ep5k/models/1000/')
            model_path = os.path.join(model_path,'model_state_%d.pth'%opt.real_per)
            dict = torch.load(model_path)
            state_dict = dict['last_state_dict_G']
            generator.load_state_dict(state_dict,strict=True)
            generator.eval()

            FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

            output_dir = model_path.replace('model_state_%d.pth'%opt.real_per,'eval_op')
            output_dir = os.path.join(output_dir,str(class_num))
            os.makedirs(output_dir,exist_ok=True)

            # Calc the number of samples to be added
            z = Variable(FloatTensor(np.random.normal(0, 1, (opt.eval_samples, opt.latent_dim))))
            labels = []
            for j in range(0,opt.eval_samples):
                labels.append(class_num)
            labels=np.array(labels)
            gen_labels = Variable(LongTensor(labels))
            output_file = os.path.join(output_dir,'syn_%d_samples.npy'%opt.eval_samples)

            syn_data = np.zeros((opt.eval_samples,3,opt.temporal_length,25,2),dtype=np.float32)
            if opt.feat_concat == 0:
                gen_imgs = generator(z, gen_labels)
            elif opt.feat_concat == 512 or opt.feat_concat == -1 or opt.feat_concat == 256:
                gen_labels_numpy = gen_labels.cpu().numpy()
                ind = 0
                out1 = np.zeros((opt.eval_samples,512),dtype=np.float32)
                if opt.feat_concat == 256:
                    out2 = np.zeros((opt.eval_samples,256),dtype=np.float32)
                for label in gen_labels_numpy:
                    feat_maps_path = '/fast-stripe/workspaces/deval/synthetic-data/hcn/feat_map/'
                    feat_maps_path = os.path.join(feat_maps_path,str(label),'feat_map.npy')
                    feat_maps = np.load(feat_maps_path)
                    out1[ind] = feat_maps
                    if opt.feat_concat == 256:
                        feat_maps_path = '/fast-stripe/workspaces/deval/synthetic-data/hcn/feat_map/'
                        feat_maps_path = os.path.join(feat_maps_path,str(label),'feat256_map.npy')
                        feat_maps = np.load(feat_maps_path)
                        out2[ind] = feat_maps
                    ind = ind + 1
                out1 = torch.from_numpy(out1).cuda()
                if opt.feat_concat == 512 or opt.feat_concat == -1:
                    gen_imgs = generator(z, gen_labels, out1)
                elif opt.feat_concat == 256:
                    out2 = torch.from_numpy(out2).cuda()
                    gen_imgs = generator(z, gen_labels, out1, out2)
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
            
            print ("Completed data saving")