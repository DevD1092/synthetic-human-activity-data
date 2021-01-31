import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class Generator(nn.Module):
    def __init__(self,latent_dim=100,img_shape=(3,25,25)):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class Generator_concat512(nn.Module):
    def __init__(self,latent_dim=100,img_shape=(3,25,25)):
        super(Generator_concat512, self).__init__()
        self.img_shape = img_shape

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.l1 = nn.Linear(latent_dim, 128)
        self.l2 = nn.Linear(128, 256)
        self.batch_norm2 = nn.BatchNorm1d(256,0.8)
        self.l3 = nn.Linear(256, 512)
        self.batch_norm3 = nn.BatchNorm1d(512,0.8)
        self.l4 = nn.Linear(1024, 1024)
        self.batch_norm4 = nn.BatchNorm1d(1024,0.8)
        self.l5 = nn.Linear(1024, int(np.prod(self.img_shape)))
        self.l6 = nn.Tanh()

    def forward(self, z, real_act_dim):
         
        x = self.leaky_relu(self.l1(z))
        x = self.leaky_relu(self.batch_norm2(self.l2(x)))
        x = self.leaky_relu(self.batch_norm3(self.l3(x)))
        #print ("L3 shape",x.shape)
        #print ("Real act dim",real_act_dim.shape)
        x = torch.cat((x,real_act_dim),dim=-1)
        x = self.leaky_relu(self.batch_norm4(self.l4(x)))
        img = self.l6(self.l5(x))
        #print ("Img 1 ",img.shape)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self,img_shape=(3,25,25)):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

class Generator1(nn.Module):
    def __init__(self,latent_dim=100,DIM=1875):
        super(Generator1, self).__init__()
        self.DIM = DIM
        preprocess = nn.Sequential(
            nn.Linear(latent_dim, 3 * 3 * 3 * DIM),
            nn.BatchNorm1d(3 * 3 * 3 * DIM),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(3 * DIM, 2 * DIM, 2, stride=2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.Sequential(nn.ConvTranspose2d(DIM, 3, 2, stride=2),nn.ReplicationPad2d((1,0,1,0)))

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        #print ("Input",input.shape)
        output = self.preprocess(input)
        #print ("Preprocess o/p",output.shape)
        output = output.view(-1, 3 * self.DIM, 3, 3)
        #print ("Preprocess1 o/p",output.shape)
        output = self.block1(output)
        #print ("Block 1 o/p",output.shape)
        output = self.block2(output)
        #print ("Block 2 o/p",output.shape)
        output = self.deconv_out(output)
        #print ("Deconv_out o/p",output.shape)
        output = self.tanh(output)
        #print ("Tanh o/p",output.shape)
        return output.view(-1, 3, 25, 25)


class Discriminator1(nn.Module):
    def __init__(self,DIM=1875):
        super(Discriminator1, self).__init__()
        self.DIM = DIM
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.DIM)
        output = self.linear(output)
        return output

class Generator2(nn.Module):
    def __init__(self,latent_dim = 100, img_shape=(3,60,25), n_classes=1):
        super(Generator2, self).__init__()

        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        #print ("Label embeddings",self.label_emb(labels).shape)
        #print ("Noise",noise.shape)
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        #print ("Gen input",gen_input.shape)
        img = self.model(gen_input)
        #print ("Img 1 ",img.shape)
        img = img.view(img.size(0), *self.img_shape)
        #print ("Img 2",img.shape)
        return img


class Discriminator2(nn.Module):
    def __init__(self,img_shape = (3,60,25), n_classes=1):
        super(Discriminator2, self).__init__()

        #self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(512, 1),nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(512, n_classes), nn.Softmax())

    #def forward(self, img,labels):
    def forward(self,img):
        # Concatenate label embedding and image to produce input
        #d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        d_in = img.view(img.size(0),-1)
        d_out = self.model(d_in)
        validity = self.adv_layer(d_out)
        c = self.aux_layer(d_out)
        return validity,c

class Generator_cgan(nn.Module):
    def __init__(self,latent_dim = 100, img_shape=(3,60,25), n_classes=1):
        super(Generator_cgan, self).__init__()

        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        #print ("Label embeddings",self.label_emb(labels).shape)
        #print ("Noise",noise.shape)
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        #print ("Gen input",gen_input.shape)
        img = self.model(gen_input)
        #print ("Img 1 ",img.shape)
        img = img.view(img.size(0), *self.img_shape)
        #print ("Img 2",img.shape)
        return img


class Discriminator_cgan(nn.Module):
    def __init__(self,img_shape = (3,60,25), n_classes=1):
        super(Discriminator_cgan, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)
        
        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )
        '''
        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )'''

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity