import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class Generator(nn.Module):
    def __init__(self,latent_dim = 100, img_shape=(3,60,25), n_classes=1):
        super(Generator, self).__init__()

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
        '''
        self.large_model = nn.Sequential(
            *block(latent_dim + n_classes, 256, normalize=False),
            *block(256, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024,1024),
            *block(1024,2048),
            nn.Linear(2048, int(np.prod(self.img_shape))),
            nn.Tanh()
        )'''

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        #print ("Label embeddings",self.label_emb(labels).shape)
        #print ("Noise",noise.shape)
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        #print ("Gen input",gen_input.shape)
        img = self.model(gen_input)
        #img = self.large_model(gen_input)
        #print ("Img 1 ",img.shape)
        img = img.view(img.size(0), *self.img_shape)
        #print ("Img 2",img.shape)
        return img

class Generator_earlyconcat(nn.Module):
    def __init__(self,latent_dim = 100, img_shape=(3,60,25), n_classes=1):
        super(Generator_earlyconcat, self).__init__()

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
        '''
        self.large_model = nn.Sequential(
            *block(latent_dim + n_classes, 256, normalize=False),
            *block(256, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024,1024),
            *block(1024,2048),
            nn.Linear(2048, int(np.prod(self.img_shape))),
            nn.Tanh()
        )'''

    def forward(self, noise, labels, real_act_dim):
        # Concatenate label embedding and image to produce input
        #print ("Label embeddings",self.label_emb(labels).shape)
        #print ("Noise",noise.shape)
        noise = noise + real_act_dim
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        #gen_input = torch.cat((gen_input,real_act_dim),dim=-1)
        #print ("Gen input",gen_input.shape)
        img = self.model(gen_input)
        #img = self.large_model(gen_input)
        #print ("Img 1 ",img.shape)
        img = img.view(img.size(0), *self.img_shape)
        #print ("Img 2",img.shape)
        return img

class Generator_concat(nn.Module):
    def __init__(self,latent_dim = 100, img_shape=(3,60,25), n_classes=1):
        super(Generator_concat, self).__init__()

        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, n_classes)
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.l1 = nn.Linear(latent_dim + n_classes, 128)
        self.l2 = nn.Linear(128, 256)
        self.batch_norm2 = nn.BatchNorm1d(256,0.8)
        self.l3 = nn.Linear(256, 512)
        self.batch_norm3 = nn.BatchNorm1d(512,0.8)
        self.l4 = nn.Linear(1024, 1024)
        self.batch_norm4 = nn.BatchNorm1d(1024,0.8)
        self.l5 = nn.Linear(1024, int(np.prod(self.img_shape)))
        self.l6 = nn.Tanh()

        '''
        self.large_model = nn.Sequential(
            *block(latent_dim + n_classes, 256, normalize=False),
            *block(256, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024,1024),
            *block(1024,2048),
            nn.Linear(2048, int(np.prod(self.img_shape))),
            nn.Tanh()
        )'''

    def forward(self, noise, labels, real_act_dim):
        # Concatenate label embedding and image to produce input
        #print ("Label embeddings",self.label_emb(labels).shape)
        #print ("Noise",noise.shape)
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        #print ("Gen input",gen_input.shape)
        x = self.leaky_relu(self.l1(gen_input))
        x = self.leaky_relu(self.batch_norm2(self.l2(x)))
        x = self.leaky_relu(self.batch_norm3(self.l3(x)))
        #print ("L3 shape",x.shape)
        #print ("Real act dim",real_act_dim.shape)
        x = torch.cat((x,real_act_dim),dim=-1)
        x = self.leaky_relu(self.batch_norm4(self.l4(x)))
        img = self.l6(self.l5(x))
        #print ("Img 1 ",img.shape)
        img = img.view(img.size(0), *self.img_shape)
        #print ("Img 2",img.shape)
        return img

class Generator_concat256(nn.Module):
    def __init__(self,latent_dim = 100, img_shape=(3,60,25), n_classes=1):
        super(Generator_concat256, self).__init__()

        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, n_classes)
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.l1 = nn.Linear(latent_dim + n_classes, 128)
        self.l2 = nn.Linear(128, 256)
        self.batch_norm2 = nn.BatchNorm1d(256,0.8)
        self.l3 = nn.Linear(512, 512)
        self.batch_norm3 = nn.BatchNorm1d(512,0.8)
        self.l4 = nn.Linear(1024, 1024)
        self.batch_norm4 = nn.BatchNorm1d(1024,0.8)
        self.l5 = nn.Linear(1024, int(np.prod(self.img_shape)))
        self.l6 = nn.Tanh()

        '''
        self.large_model = nn.Sequential(
            *block(latent_dim + n_classes, 256, normalize=False),
            *block(256, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024,1024),
            *block(1024,2048),
            nn.Linear(2048, int(np.prod(self.img_shape))),
            nn.Tanh()
        )'''

    def forward(self, noise, labels, real_act_dim, real_act_dim_256):
        # Concatenate label embedding and image to produce input
        #print ("Label embeddings",self.label_emb(labels).shape)
        #print ("Noise",noise.shape)
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        #print ("Gen input",gen_input.shape)
        x = self.leaky_relu(self.l1(gen_input))
        x = self.leaky_relu(self.batch_norm2(self.l2(x)))
        x = torch.cat((x, real_act_dim_256), dim=-1)
        x = self.leaky_relu(self.batch_norm3(self.l3(x)))
        #print ("L3 shape",x.shape)
        #print ("Real act dim",real_act_dim.shape)
        x = torch.cat((x,real_act_dim),dim=-1)
        x = self.leaky_relu(self.batch_norm4(self.l4(x)))
        img = self.l6(self.l5(x))
        #print ("Img 1 ",img.shape)
        img = img.view(img.size(0), *self.img_shape)
        #print ("Img 2",img.shape)
        return img

class Discriminator(nn.Module):
    def __init__(self,img_shape = (3,60,25), n_classes=1):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)
        
        self.model = nn.Sequential(
            nn.Linear(n_classes+int(np.prod(img_shape)), 512), #n_classes + 
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
        self.large_model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.Dropout(0.25),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.25),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.Dropout(0.25),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )'''

    def forward(self, img, labels): #,labels
        # Concatenate label embedding and image to produce input
        #print ("Img size",img.size())
        #print ("Labels size",labels.size())
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        #img = img.view(img.size(0),-1)
        validity = self.model(d_in) #d_in
        #validity = self.large_model(d_in)
        return validity

class Generator_cnn(nn.Module):
    # initializers
    def __init__(self, latent_dim=100, DIM=4500,classes=49):
        super(Generator_cnn, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(latent_dim, DIM*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(DIM*2)
        self.deconv1_2 = nn.ConvTranspose2d(classes, DIM*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(DIM*2)
        self.deconv2 = nn.ConvTranspose2d(DIM*4, DIM*2, 4, (1,4), (1,2), output_padding=(0,1))
        self.deconv2_bn = nn.BatchNorm2d(DIM*2)
        self.deconv3 = nn.ConvTranspose2d(DIM*2, DIM, 4, (2,2), output_padding=(0,1))
        self.deconv3_bn = nn.BatchNorm2d(DIM)
        self.deconv4 = nn.ConvTranspose2d(DIM, 3, 4, (2,2), (1,0), output_padding=(1,0))

        #Linear tail
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.linear_tailmodel = nn.Sequential(
            *block(25*3*60,1024),            
            nn.Linear(1024,25*3*60)
        )
        #self.linear_tailmodel = nn.Sequential(            
        #    nn.Linear(1024,num_point*3*window_size)
        #)
        self.img_shape = (3,25,60) # C,V,T


    # forward method
    def forward(self, input, label):
        #print ("Gen input",input.shape)
        #print ("Gen label",label.shape)
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        #print ("Deconv 1_1",x.shape)
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        #print ("Devonc 1_2",y.shape)
        x = torch.cat([x, y], 1)
        #print ("Concat",x.shape)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        #print ("Devonc 2",x.shape)
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        #print ("Devonc 3",x.shape)
        x = self.deconv4(x)
        x = x.view(x.size(0),-1)
        x = F.tanh(self.linear_tailmodel(x))
        #x = F.tanh(self.deconv4(x))
        #print ("Devonc 4",x.shape)
        x = x.view(x.size(0), *self.img_shape)
        #print ("Final x Generator",x.size())
        return x

class Discriminator_cnn(nn.Module):
    # initializers
    def __init__(self, d=4500, classes=49):
        super(Discriminator_cnn, self).__init__()
        self.conv1_1 = nn.Conv2d(3, d//2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(classes, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, 1, (3,7), 1, 0)

    def forward(self, input, label):
        #print (input.shape)
        label = label.permute(0,1,3,2)
        #print (label.shape)
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        #print (x.shape)
        #print (y.shape)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        #print ("Conv2_bn",x.shape)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        #print ("Conv3_bn",x.shape)
        x = F.sigmoid(self.conv4(x))
        #x = self.conv4(x)
        #print ("Sigmoid",x.shape)
        return x