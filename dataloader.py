import numpy as np
import sys
import os
import torch
import scipy.misc
import pickle
import matplotlib.pyplot as plt
import random

from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset
from interpolate import valid_crop_resize_multi_data
from numpy import inf
from collections import Counter

joint_seq = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
        (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
        (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
        (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
        (22, 23), (23, 8), (24, 25), (25, 12)]

def reshape_data_vacnn(data):
    data = np.transpose(data,(1,3,2,0))
    data = np.reshape(data,(data.shape[0],data.shape[1],data.shape[2]*data.shape[3]))
    data = np.reshape(data,(data.shape[0],data.shape[1]*data.shape[2]))
    data = np.expand_dims(data,axis=0)
    return data

def calc_num_frames_vacnn(data,cvm):

    data = reshape_data_vacnn(data)

    zero_row = []
    cvm = int(cvm)
    ske_joint = np.squeeze(data,axis=0)
    #print (ske_joint.shape)
    for i in range(len(ske_joint)):
        if (ske_joint[i, :] == np.zeros((1, cvm))).all():
            zero_row.append(i)
    ske_joint = np.delete(ske_joint, zero_row, axis=0)
    
    return (ske_joint.shape[0])

def scale(_data):
    data_scaled = _data.astype('float32')
    data_max = np.max(data_scaled)
    data_min = np.min(data_scaled)
    data_scaled = (_data-data_min)/(data_max-data_min)
    return data_scaled, data_max, data_min


# descale generated data
def descale(data, data_max, data_min):
    data_descaled = data*(data_max-data_min)+data_min
    return data_descaled

def convert_to_rgb_vacnn(data,max_val,min_val,cvm):
    zero_row = []
    cvm = int(cvm)
    ske_joint = np.squeeze(data,axis=0)
    #print (ske_joint.shape)
    for i in range(len(ske_joint)):
        if (ske_joint[i, :] == np.zeros((1, cvm))).all():
            zero_row.append(i)
    ske_joint = np.delete(ske_joint, zero_row, axis=0)
    
    if (ske_joint[:, 0:cvm//2] == np.zeros((ske_joint.shape[0], cvm//2))).all():
        ske_joint = np.delete(ske_joint, range(cvm//2), axis=1)
    elif (ske_joint[:, cvm//2:cvm] == np.zeros((ske_joint.shape[0], cvm//2))).all():
        ske_joint = np.delete(ske_joint, range(cvm//2, cvm), axis=1)
    
    # Convert to RGB
    ske_joint =  255 * (ske_joint - min_val) / (max_val - min_val)
    rgb_ske = np.reshape(ske_joint, (ske_joint.shape[0], ske_joint.shape[1] //3, 3))
    if rgb_ske.shape[0]==0:
        rgb_ske = np.zeros([224,224,3],dtype=np.float32)
    else:
        rgb_ske = scipy.misc.imresize(rgb_ske, (224, 224)).astype(np.float32)
    #rgb_ske = np.array(Image.fromarray(rgb_ske.astype('uint8')).resize((224,224))).astype(np.float32)
    rgb_ske = center(rgb_ske)
    rgb_ske = np.transpose(rgb_ske, [1, 0, 2])
    rgb_ske = np.transpose(rgb_ske, [2,1,0])
    return rgb_ske

def convert_to_rgb_interpolate(data,max_val,min_val,cvm,center_flag):
    zero_row = []
    cvm = int(cvm)
    ske_joint = np.squeeze(data,axis=0)
    for i in range(len(ske_joint)):
        if (ske_joint[i, :] == np.zeros((1, cvm))).all():
            zero_row.append(i)
    ske_joint = np.delete(ske_joint, zero_row, axis=0)
    
    if (ske_joint[:, 0:cvm//2] == np.zeros((ske_joint.shape[0], cvm//2))).all():
        ske_joint = np.delete(ske_joint, range(cvm//2), axis=1)
    elif (ske_joint[:, cvm//2:cvm] == np.zeros((ske_joint.shape[0], cvm//2))).all():
        ske_joint = np.delete(ske_joint, range(cvm//2, cvm), axis=1)

    ske_joint =  255 * (ske_joint - min_val) / (max_val - min_val)
    rgb_ske = np.reshape(ske_joint, (ske_joint.shape[0], ske_joint.shape[1] //3, 3))
    rgb_ske = np.transpose(rgb_ske, [1, 0, 2])
    rgb_ske = np.transpose(rgb_ske, [2,1,0])
    return rgb_ske

def normalize_data(data,max_val,min_val,range_num):
    if range_num == 0:
        data = 255*(data-min_val)/(max_val-min_val)
        data = center1(data)
    else:
        data = 2*(data-min_val)/(max_val-min_val) - 1
    return data

def vis_3dske(data_3d,num_frames_3d,save_dir,sample,normalize_flag,spherical_flag):

    if spherical_flag == 1:
        num_joints = data_3d.shape[2]
        data_sph = np.zeros((3,num_frames_3d,num_joints),dtype=np.float32)
        for i in range(num_frames_3d):
            for j in range(num_joints):
                az, el, r = data_3d[0,i,j], data_3d[1,i,j], data_3d[2,i,j]
                #print (x,y,z)
                x, y, z = sph2cart(az,el,r)
                #print (az,el,r)
                #print (x,y,z)
                data_sph[0,:,:] = x
                data_sph[1,:,:] = y
                data_sph[2,:,:] = z
        data_per1 = data_sph
    #elif normalize_flag == 1:
    #    max_val,min_val = 5.826573,-4.9881773
    #    data_3d = descale(data_3d,data_max=max_val,data_min=min_val)
    #    data_per1 = data_3d
    else:
        data_per1 = data_3d

    data_per1 = np.expand_dims(data_per1,axis=3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for frame in range(num_frames_3d):
        world_head_bones_per1_x = []
        world_head_bones_per1_y = []
        world_head_bones_per1_z = []

        world_head_per1 = data_per1[:,frame,:,:]
        world_head_per1 = np.squeeze(world_head_per1,axis=2)
        world_head_per1 = np.transpose(world_head_per1,(1,0))

        world_head_per1_x = world_head_per1[:,0]
        world_head_per1_y = world_head_per1[:,1]
        world_head_per1_z = world_head_per1[:,2]

        # All bones
        for i in range(len(joint_seq)):
            #print (world_head_x[joint_seq[i][0]],world_head_x[joint_seq[i][1]])
            world_head_bones_per1_x.append([world_head_per1_x[joint_seq[i][0]-1],world_head_per1_x[joint_seq[i][1]-1]])
            world_head_bones_per1_y.append([world_head_per1_y[joint_seq[i][0]-1],world_head_per1_y[joint_seq[i][1]-1]])
            world_head_bones_per1_z.append([world_head_per1_z[joint_seq[i][0]-1],world_head_per1_z[joint_seq[i][1]-1]])

        world_head_bones_per1_x = np.array(world_head_bones_per1_x)
        world_head_bones_per1_y = np.array(world_head_bones_per1_y)
        world_head_bones_per1_z = np.array(world_head_bones_per1_z)

        plt.cla()
        
        for i in range(len(joint_seq)):
            ax.plot(world_head_bones_per1_z[i],world_head_bones_per1_x[i],world_head_bones_per1_y[i],color='blue')
        
        ax.scatter(world_head_per1_z,world_head_per1_x,world_head_per1_y,s=25,label='True Position')

        #if normalize_flag == 1:
        #    ax.set_xlim(-0.5,0.5)
        #    ax.set_ylim(-0.5,0.5)
        #    ax.set_zlim(-0.5,0.5)
        #else:
        ax.set_xlim(-1,4)
        ax.set_ylim(-1,2)
        ax.set_zlim(-1,2)
        #if flag_rgb_to_3d == 1:
        #    ax2.set_xlim(0,1)
        #    ax2.set_ylim(0,1)
        #    ax2.set_zlim(0,1)
        ax.set_title('Bones, frame={}'.format(frame))
        plt.ioff()
        plt.savefig(save_dir + "/sam%05dim%03d.png" % (sample,frame))
    plt.close('all')

def center(rgb):
    rgb[:,:,0] -= 110
    rgb[:,:,1] -= 110
    rgb[:,:,2] -= 110
    return rgb

def center1(rgb):
    rgb[0,:,:] -= 110
    rgb[1,:,:] -= 110
    rgb[2,:,:] -= 110
    return rgb

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

# DCGAN,WGAN_GP single class dataloader
class NTURGBDData(Dataset):

    def __init__(self,temporal_length,temporal_pattern,gpu_id,dataset,split,normalize_flag,centering_flag,spherical_flag,class_num,real_per,num_person,set_,transform,expt):
        self.temporal_length = temporal_length
        self.labels = None
        self.temporal_pattern = temporal_pattern
        self.set_=set_
        self.split = split
        self.gpu_id = gpu_id
        self.dataset = dataset
        self.expt = expt
        self.normalize = normalize_flag
        self.centering = centering_flag
        self.spherical = spherical_flag
        self.class_num = class_num
        self.real_per = real_per
        self.num_person = num_person
        self.transform = transform
        
        print ("Loading ",self.dataset, "data")
        if self.dataset == 'vacnn':
            self.data_path_3d = "/fast-stripe/datasets/nyu_action/ntu_3d_data_vacnn/"
            self.deid_path = "/fast-stripe/datasets/nyu_action/ntu_3d_data_vacnn/"
        elif self.dataset == 'dgnn':
            self.data_path_3d = "/fast-stripe/datasets/nyu_action/ntu_3d_data_dgnn/"
            self.deid_path = "/fast-stripe/datasets/nyu_action/deid_multiperson_data/"
        else:
            self.data_path_3d = "/fast-stripe/datasets/nyu_action/ntu_3d_data/"
            self.deid_path = "/fast-stripe/datasets/nyu_action/deid_multiperson_data/"

        data_3d = set_+"_"+split+"_"+"3ddata.npy"
        labels_3d = set_+"_"+split+"_"+"label_3ddata.pkl"
        num_frames_3d = set_+"_"+split+"_"+"num_frame_3ddata.npy"
        data_bone_3d = set_+"_"+split+"_"+"data_bone.npy"
        self.data_bone_3d_path = os.path.join(self.data_path_3d,data_bone_3d)
        self.data_3d_path = os.path.join(self.data_path_3d,data_3d)
        self.labels_3d_path = os.path.join(self.data_path_3d,labels_3d)
        self.num_frames_3d_path = os.path.join(self.data_path_3d,num_frames_3d)

        with open(self.labels_3d_path,"rb") as f:
            self.video_name,self.labels_3d = pickle.load(f)
        f.close()

        self.data_3d = np.load(self.data_3d_path)
        self.num_frames_3d = np.load(self.num_frames_3d_path)

        if self.dataset == 'vacnn':
            self.video_name = np.array(self.video_name)
            self.video_name = self.video_name.astype('U25')

        print ("Loading dataset",self.dataset)

        #============================================ Single-person ACTS ================================#
        ignore_samples_path = '/fast-stripe/datasets/nyu_action/samples_with_missing_skeleton.txt'
        with open(ignore_samples_path, 'r') as f:
            ignored_samples = [line.strip() for line in f.readlines()]
        f.close()

        if self.temporal_pattern == 'interpolate':
            tot_samples = 0
            for i in range(len(self.labels_3d)):
                if (self.labels_3d[i] == self.class_num): # <=48 : Single person acts ; 42 - Fall ; 26 - Jump
                    if self.dataset != 'vacnn':
                        video_name = self.video_name[i].replace(".skeleton","")
                    else:
                        video_name = self.video_name[i]
                    if video_name not in ignored_samples:
                        tot_samples = tot_samples+1

            #print ("Total single activity samples",tot_samples)
            self.data_3d_inter = np.zeros((tot_samples,3,300,25,2),dtype=np.float32)
            self.num_frames_3d_inter = np.zeros((tot_samples),dtype=np.int32)
            self.video_name_inter = []
            self.labels_3d_inter = []
            inter_ind = 0
            sample_indices = []
            for i in range(len(self.labels_3d)):
                #print (self.labels_3d[i])
                if (self.labels_3d[i] == self.class_num):
                    if self.dataset != 'vacnn':
                        video_name = self.video_name[i].replace(".skeleton","")
                    else:
                        video_name = self.video_name[i]
                    if video_name not in ignored_samples:
                        self.data_3d_inter[inter_ind] = self.data_3d[i]
                        #print (self.data_3d_inter[inter_ind])
                        self.num_frames_3d_inter[inter_ind] = self.num_frames_3d[i]
                        #print (self.num_frames_3d_inter)
                        self.video_name_inter.append(self.video_name[i])
                        self.labels_3d_inter.append(self.labels_3d[i])
                        sample_indices.append(i)
                        inter_ind = inter_ind + 1
            
            # Shuffle the data for real percentage
            if self.real_per != 100:
                shuffle_samples = int((self.real_per/100)*tot_samples)
                #print (tot_samples,shuffle_samples)
                shuffle_pattern = random.sample(range(0,tot_samples),shuffle_samples)
                self.data_3d_shuffle = np.zeros((shuffle_samples,3,300,25,2),dtype=np.float32)
                self.num_frames_3d_shuffle = np.zeros((shuffle_samples),dtype=np.int32)
                self.video_name_shuffle = []
                self.labels_3d_shuffle = []
                sample_indices_shuffled = np.zeros((shuffle_samples),dtype=np.int32)
                per_ind = 0

                for shuffle_ind in shuffle_pattern:
                    self.data_3d_shuffle[per_ind] = self.data_3d_inter[shuffle_ind]
                    self.num_frames_3d_shuffle[per_ind] = self.num_frames_3d_inter[shuffle_ind]
                    self.video_name_shuffle.append(self.video_name_inter[shuffle_ind])
                    self.labels_3d_shuffle.append(self.labels_3d_inter[shuffle_ind])
                    sample_indices_shuffled[per_ind] = sample_indices[shuffle_ind]
                    per_ind = per_ind + 1
                #print (sample_indices_shuffled)
                # Save the indices
                ind_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp',str(self.class_num),'real_data')
                os.makedirs(ind_path,exist_ok=True)
                ind_path = os.path.join(ind_path,'ind_%d.npy'%self.real_per)
                np.save(ind_path,sample_indices_shuffled)

                self.data_3d = self.data_3d_shuffle
                self.num_frames_3d = self.num_frames_3d_shuffle
                self.video_name = self.video_name_shuffle
                self.labels_3d = self.labels_3d_shuffle

            else:
                self.data_3d = self.data_3d_inter
                self.num_frames_3d = self.num_frames_3d_inter
                self.video_name = self.video_name_inter
                self.labels_3d = self.labels_3d_inter

            #print (self.data_3d_inter.shape[0],self.num_frames_3d_inter.shape[0])
            #print (self.data_3d.shape[0],self.num_frames_3d.shape[0],len(self.video_name))
            #labels_count = Counter(self.labels_3d)

        print ("Loaded the %s set"%set_,"Total samples",len(self.video_name))   

    def __len__(self):
        if (self.expt == 'check'):
            return (len(self.video_name[0:10]))
        else:
            return (len(self.video_name))
    
    def __getitem__(self,id):

        # Get the labels
        self.labels = self.labels_3d[id]
        self.video = self.video_name[id]
        self.num_frames = self.num_frames_3d[id]
        if self.dataset != 'vacnn':
            self.video = self.video.replace(".skeleton","")

        # Get the data
        self.data = self.data_3d[id]
        self.num_frames = calc_num_frames_vacnn(self.data,150)

        # Get the deid data & convert nans to 0.0
        if self.temporal_pattern == 'interpolate':
            if self.centering == 1:
                origin_data = self.data[:,:,1,0]
                self.data = self.data - origin_data[:,:,None,None]
                #print ("Before",self.data)
            p_interval = [0.5,1]
            p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
            #print (self.data.shape,self.num_frames)
            self.data = valid_crop_resize_multi_data(self.data,self.num_frames,p_interval,p,self.temporal_length)
            if self.num_person == 1:
                self.data = self.data[:,:,:,0:1]
                self.data = np.squeeze(self.data)
            elif self.num_person == 2:
                data_per1 = self.data[:,:,:,0:1]
                data_per1 = np.squeeze(data_per1)
                data_per2 = self.data[:,:,:,1:2]
                data_per2 = np.squeeze(data_per2)
                self.data = np.concatenate((data_per1,data_per2),axis=-1)
                #print (self.data.shape)
                #print ("After",self.data)
            self.data = np.nan_to_num(self.data)
            self.data[self.data == -inf] = 0
            if self.normalize == 1:
                #max_val,min_val = 5.18858098984,-5.28981208801
                max_val,min_val = 5.826573,-4.9881773
                self.data = normalize_data(self.data,max_val,min_val,-1)
            if self.spherical == 1:
                num_joints = self.data.shape[2]
                data_sph = np.zeros((3,self.temporal_length,num_joints),dtype=np.float32)
                for i in range(self.temporal_length):
                    for j in range(num_joints):
                        x,y,z = self.data[0,i,j], self.data[1,i,j], self.data[2,i,j]
                        #print (x,y,z)
                        az, el, r = cart2sph(x,y,z)
                        #print (az,el,r)
                        #print (x,y,z)
                        data_sph[0,:,:] = az
                        data_sph[1,:,:] = el
                        data_sph[2,:,:] = r
                self.data = data_sph
            #print (self.data.shape)
            #self.data = reshape_data_vacnn(self.data)
            #rgb_ske = convert_to_rgb_interpolate(self.data,max_val,min_val,150,0)
            #print (rgb_ske.shape)
            if self.transform is not None:
                data_tensor =  torch.from_numpy(self.data)
                self.data = self.transform(data_tensor)
                #print (self.data.shape)
            return self.data,0
        else:
            max_val,min_val = 5.18858098984,-5.28981208801
            self.data = reshape_data_vacnn(self.data)
            rgb_ske = convert_to_rgb_vacnn(self.data,max_val,min_val,150) # C*V*M = 150
            return rgb_ske,self.labels

## CGAN/ACGAN/WGAN_GP with all the classes data loader
class NTURGBDData_full(Dataset):

    def __init__(self,temporal_length,temporal_pattern,gpu_id,dataset,split,normalize_flag,centering_flag,spherical_flag,real_per,num_person,set_,transform,expt):
        self.temporal_length = temporal_length
        self.labels = None
        self.temporal_pattern = temporal_pattern
        self.set_=set_
        self.split = split
        self.gpu_id = gpu_id
        self.dataset = dataset
        self.expt = expt
        self.normalize = normalize_flag
        self.centering = centering_flag
        self.spherical = spherical_flag
        self.real_per = real_per
        self.num_person = num_person
        self.transform = transform

        if self.dataset == 'vacnn':
            self.data_path_3d = "/fast-stripe/datasets/nyu_action/ntu_3d_data_vacnn/"
            self.deid_path = "/fast-stripe/datasets/nyu_action/ntu_3d_data_vacnn/"
        elif self.dataset == 'dgnn':
            self.data_path_3d = "/fast-stripe/datasets/nyu_action/ntu_3d_data_dgnn/"
            self.deid_path = "/fast-stripe/datasets/nyu_action/deid_multiperson_data/"
        else:
            self.data_path_3d = "/fast-stripe/datasets/nyu_action/ntu_3d_data/"
            self.deid_path = "/fast-stripe/datasets/nyu_action/deid_multiperson_data/"

        data_3d = set_+"_"+split+"_"+"3ddata.npy"
        labels_3d = set_+"_"+split+"_"+"label_3ddata.pkl"
        num_frames_3d = set_+"_"+split+"_"+"num_frame_3ddata.npy"
        data_bone_3d = set_+"_"+split+"_"+"data_bone.npy"
        self.data_bone_3d_path = os.path.join(self.data_path_3d,data_bone_3d)
        self.data_3d_path = os.path.join(self.data_path_3d,data_3d)
        self.labels_3d_path = os.path.join(self.data_path_3d,labels_3d)
        self.num_frames_3d_path = os.path.join(self.data_path_3d,num_frames_3d)

        with open(self.labels_3d_path,"rb") as f:
            self.video_name,self.labels_3d = pickle.load(f)
        f.close()

        self.data_3d = np.load(self.data_3d_path)
        self.num_frames_3d = np.load(self.num_frames_3d_path)

        if self.dataset == 'vacnn':
            self.video_name = np.array(self.video_name)
            self.video_name = self.video_name.astype('U25')
        
        print ("Loading dataset ",self.dataset)

        #============================================ Single-person ACTS ================================#
        ignore_samples_path = '/fast-stripe/datasets/nyu_action/samples_with_missing_skeleton.txt'
        with open(ignore_samples_path, 'r') as f:
            ignored_samples = [line.strip() for line in f.readlines()]
        f.close()

        if self.temporal_pattern == 'interpolate':
            tot_samples = 0
            for i in range(len(self.labels_3d)):
                if (self.labels_3d[i] <= 48): # <=48 : Single person acts ; 42 - Fall ; 26 - Jump
                    if self.dataset != 'vacnn':
                        video_name = self.video_name[i].replace(".skeleton","")
                    else:
                        video_name = self.video_name[i]
                    if video_name not in ignored_samples:
                        tot_samples = tot_samples+1

            #print ("Total single activity samples",tot_samples)
            self.data_3d_inter = np.zeros((tot_samples,3,300,25,2),dtype=np.float32)
            self.num_frames_3d_inter = np.zeros((tot_samples),dtype=np.int32)
            self.video_name_inter = []
            self.labels_3d_inter = []
            inter_ind = 0
            sample_indices = []
            for i in range(len(self.labels_3d)):
                #print (self.labels_3d[i])
                if (self.labels_3d[i] <= 48):
                    if self.dataset != 'vacnn':
                        video_name = self.video_name[i].replace(".skeleton","")
                    else:
                        video_name = self.video_name[i]
                    if video_name not in ignored_samples:
                        self.data_3d_inter[inter_ind] = self.data_3d[i]
                        #print (self.data_3d_inter[inter_ind])
                        self.num_frames_3d_inter[inter_ind] = self.num_frames_3d[i]
                        #print (self.num_frames_3d_inter)
                        self.video_name_inter.append(self.video_name[i])
                        self.labels_3d_inter.append(self.labels_3d[i])
                        sample_indices.append(i)
                        inter_ind = inter_ind + 1

            if self.real_per != 100:
                shuffle_samples = int((self.real_per/100)*tot_samples)
                #print (tot_samples,shuffle_samples)
                shuffle_pattern = []
                if self.num_person == 1:
                    numclasses = 49 #TODO:Number of classes change
                    shuffle_sample_per_class = shuffle_samples//numclasses 
                elif self.num_person == 2:
                    numclasses = 60
                    shuffle_sample_per_class = shuffle_samples//numclasses
                for class_num in range(0,numclasses):
                    class_count = 0
                    for ind,label in enumerate(self.labels_3d_inter):
                        if (label == class_num):
                            class_count = class_count + 1
                            shuffle_pattern.append(ind)
                        if class_count == shuffle_sample_per_class:
                            break

                #print ("Shuffle samples",len(shuffle_pattern))
                #shuffle_pattern = random.sample(range(0,tot_samples),shuffle_samples)
                self.data_3d_shuffle = np.zeros((shuffle_samples,3,300,25,2),dtype=np.float32)
                self.num_frames_3d_shuffle = np.zeros((shuffle_samples),dtype=np.int32)
                self.video_name_shuffle = []
                self.labels_3d_shuffle = []
                per_ind = 0

                for shuffle_ind in shuffle_pattern:
                    self.data_3d_shuffle[per_ind] = self.data_3d_inter[shuffle_ind]
                    self.num_frames_3d_shuffle[per_ind] = self.num_frames_3d_inter[shuffle_ind]
                    self.video_name_shuffle.append(self.video_name_inter[shuffle_ind])
                    self.labels_3d_shuffle.append(self.labels_3d_inter[shuffle_ind])
                    per_ind = per_ind + 1

                self.data_3d = self.data_3d_shuffle
                self.num_frames_3d = self.num_frames_3d_shuffle
                self.video_name = self.video_name_shuffle
                self.labels_3d = self.labels_3d_shuffle

            else:
                self.data_3d = self.data_3d_inter
                self.num_frames_3d = self.num_frames_3d_inter
                self.video_name = self.video_name_inter
                self.labels_3d = self.labels_3d_inter

            #print (self.data_3d_inter.shape[0],self.num_frames_3d_inter.shape[0])
            #print (self.data_3d.shape[0],self.num_frames_3d.shape[0],len(self.video_name))
            labels_count = Counter(self.labels_3d)
            print ("Labels count",labels_count)

        print ("Loaded the %s set"%set_,"Total samples",len(self.video_name))

    def __len__(self):
        if (self.expt == 'check'):
            return (len(self.video_name[0:10]))
        else:
            return (len(self.video_name))
    
    def __getitem__(self,id):

        # Get the labels
        self.labels = self.labels_3d[id]
        self.video = self.video_name[id]
        self.num_frames = self.num_frames_3d[id]
        if self.dataset != 'vacnn':
            self.video = self.video.replace(".skeleton","")

        # Get the data
        self.data = self.data_3d[id]
        self.data = np.nan_to_num(self.data)
        self.data[self.data == -inf] = 0
        if self.dataset == 'vacnn':
            self.num_frames = calc_num_frames_vacnn(self.data,150)

        # Get the deid data & convert nans to 0.0
        if self.temporal_pattern == 'interpolate':
            if self.centering == 1:
                origin_data = self.data[:,:,1,0]
                self.data = self.data - origin_data[:,:,None,None]
                #print ("Before",self.data)
            p_interval = [0.5,1]
            p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
            #print (self.data.shape,self.num_frames)
            self.data = valid_crop_resize_multi_data(self.data,self.num_frames,p_interval,p,self.temporal_length)
            if self.num_person == 1:
                #self.data = self.data[:,:,:,0:1]
                #self.data = np.squeeze(self.data)
                self.data = self.data
            elif self.num_person == 2:
                data_per1 = self.data[:,:,:,0:1]
                data_per1 = np.squeeze(data_per1)
                data_per2 = self.data[:,:,:,1:2]
                data_per2 = np.squeeze(data_per2)
                self.data = np.concatenate((data_per1,data_per2),axis=-1)
                #print (self.data.shape)
                #print ("After",self.data)
            #self.data[self.data == -inf] = 0
            if self.normalize == 1:
                #print ("Normalize")
                #max_val,min_val = 5.18858098984,-5.28981208801
                max_val,min_val = 5.826573,-4.9881773
                self.data = normalize_data(self.data,max_val,min_val,0)
            if self.spherical == 1:
                num_joints = self.data.shape[2]
                data_sph = np.zeros((3,self.temporal_length,num_joints),dtype=np.float32)
                for i in range(self.temporal_length):
                    for j in range(num_joints):
                        x,y,z = self.data[0,i,j], self.data[1,i,j], self.data[2,i,j]
                        #print (x,y,z)
                        az, el, r = cart2sph(x,y,z)
                        #print (az,el,r)
                        #print (x,y,z)
                        data_sph[0,:,:] = az
                        data_sph[1,:,:] = el
                        data_sph[2,:,:] = r
                self.data = data_sph
            #print (self.data.shape)
            #self.data = reshape_data_vacnn(self.data)
            #rgb_ske = convert_to_rgb_interpolate(self.data,max_val,min_val,150,0)
            #print (rgb_ske.shape)
            if self.transform is not None:
                data_tensor =  torch.from_numpy(self.data)
                self.data = self.transform(data_tensor)
                #print (self.data.shape)
            return self.data,self.labels
        else:
            max_val,min_val = 5.826573,-4.9881773
            self.data = reshape_data_vacnn(self.data)
            rgb_ske = convert_to_rgb_vacnn(self.data,max_val,min_val,150) # C*V*M = 150
            return rgb_ske,self.labels

## HCN normal dataloader
class NTURGBDData1(Dataset):

    def __init__(self,temporal_length,temporal_pattern,gpu_id,dataset,split,normalize_flag,centering_flag,spherical_flag,syn_per,only_syn,real_per,set_,expt):
        self.temporal_length = temporal_length
        self.labels = None
        self.temporal_pattern = temporal_pattern
        self.set_=set_
        self.split = split
        self.gpu_id = gpu_id
        self.dataset = dataset
        self.expt = expt
        self.normalize = normalize_flag
        self.centering = centering_flag
        self.spherical = spherical_flag
        self.syn_per = syn_per
        self.only_syn = only_syn
        self.real_per = real_per

        if self.dataset == 'vacnn':
            self.data_path_3d = "/fast-stripe/datasets/nyu_action/ntu_3d_data_vacnn/"
            self.deid_path = "/fast-stripe/datasets/nyu_action/ntu_3d_data_vacnn/"
        elif self.dataset == 'dgnn':
            self.data_path_3d = "/fast-stripe/datasets/nyu_action/ntu_3d_data_dgnn/"
            self.deid_path = "/fast-stripe/datasets/nyu_action/deid_multiperson_data/"
        else:
            self.data_path_3d = "/fast-stripe/datasets/nyu_action/ntu_3d_data/"
            self.deid_path = "/fast-stripe/datasets/nyu_action/deid_multiperson_data/"

        data_3d = set_+"_"+split+"_"+"3ddata.npy"
        labels_3d = set_+"_"+split+"_"+"label_3ddata.pkl"
        num_frames_3d = set_+"_"+split+"_"+"num_frame_3ddata.npy"
        data_bone_3d = set_+"_"+split+"_"+"data_bone.npy"
        self.data_bone_3d_path = os.path.join(self.data_path_3d,data_bone_3d)
        self.data_3d_path = os.path.join(self.data_path_3d,data_3d)
        self.labels_3d_path = os.path.join(self.data_path_3d,labels_3d)
        self.num_frames_3d_path = os.path.join(self.data_path_3d,num_frames_3d)

        with open(self.labels_3d_path,"rb") as f:
            self.video_name,self.labels_3d = pickle.load(f)
        f.close()

        self.data_3d = np.load(self.data_3d_path)
        self.num_frames_3d = np.load(self.num_frames_3d_path)

        #============================================ Single-person Real ACTS ================================#
        ignore_samples_path = '/fast-stripe/datasets/nyu_action/samples_with_missing_skeleton.txt'
        with open(ignore_samples_path, 'r') as f:
            ignored_samples = [line.strip() for line in f.readlines()]
        f.close()

        if self.temporal_pattern == 'interpolate':
            if self.only_syn == 0:
                tot_samples = 0
                for i in range(len(self.labels_3d)):
                    if (self.labels_3d[i] <= 48): # <=48 : Single person acts ; 42 - Fall ; 26 - Jump
                        video_name = self.video_name[i].replace(".skeleton","")
                        if video_name not in ignored_samples:
                            tot_samples = tot_samples+1
                #print ("Total single activity samples",tot_samples)
                self.data_3d_inter = np.zeros((tot_samples,3,300,25,2),dtype=np.float32)
                self.num_frames_3d_inter = np.zeros((tot_samples),dtype=np.int32)
                self.video_name_inter = []
                self.labels_3d_inter = []
                self.type_label = [] # 1-Real data ; 0-Synthetic data
                inter_ind = 0
                for i in range(len(self.labels_3d)):
                    #print (self.labels_3d[i])
                    if (self.labels_3d[i] <= 48): # <=48 : Single person acts ; 42 - Fall ; 26 - Jump
                        video_name = self.video_name[i].replace(".skeleton","")
                        if video_name not in ignored_samples:
                            self.data_3d_inter[inter_ind] = self.data_3d[i]
                            #print (self.data_3d_inter[inter_ind])
                            self.num_frames_3d_inter[inter_ind] = self.num_frames_3d[i]
                            #print (self.num_frames_3d_inter)
                            self.video_name_inter.append(self.video_name[i])
                            self.labels_3d_inter.append(self.labels_3d[i])
                            self.type_label.append(1)
                            inter_ind = inter_ind + 1

                if self.real_per != 100:
                    shuffle_samples = int((self.real_per/100)*tot_samples)
                    #print (tot_samples,shuffle_samples)
                    shuffle_pattern = []
                    # if self.num_person == 1:
                    numclasses = 49 #TODO:Number of classes change
                    shuffle_sample_per_class = shuffle_samples//numclasses 
                    #elif self.num_person == 2:
                    #    numclasses = 60
                    #    shuffle_sample_per_class = shuffle_samples//numclasses
                    for class_num in range(0,numclasses):
                        class_count = 0
                        for ind,label in enumerate(self.labels_3d_inter):
                            if (label == class_num):
                                class_count = class_count + 1
                                shuffle_pattern.append(ind)
                            if class_count == shuffle_sample_per_class:
                                break
                    #shuffle_pattern = random.sample(range(0,tot_samples),shuffle_samples)
                    self.data_3d_shuffle = np.zeros((shuffle_samples,3,300,25,2),dtype=np.float32)
                    self.num_frames_3d_shuffle = np.zeros((shuffle_samples),dtype=np.int32)
                    self.video_name_shuffle = []
                    self.labels_3d_shuffle = []
                    per_ind = 0

                    for shuffle_ind in shuffle_pattern:
                        self.data_3d_shuffle[per_ind] = self.data_3d_inter[shuffle_ind]
                        self.num_frames_3d_shuffle[per_ind] = self.num_frames_3d_inter[shuffle_ind]
                        self.video_name_shuffle.append(self.video_name_inter[shuffle_ind])
                        self.labels_3d_shuffle.append(self.labels_3d_inter[shuffle_ind])
                        per_ind = per_ind + 1

                    self.data_3d = self.data_3d_shuffle
                    self.num_frames_3d = self.num_frames_3d_shuffle
                    self.video_name = self.video_name_shuffle
                    self.labels_3d = self.labels_3d_shuffle

                else:
                    self.data_3d = self.data_3d_inter
                    self.num_frames_3d = self.num_frames_3d_inter
                    self.video_name = self.video_name_inter
                    self.labels_3d = self.labels_3d_inter               

            #============================================ Single-person Synthetic ACTS ================================#
            if self.set_ == 'train' and self.syn_per != 0:
                syn_samples = 0

                for class_num in range(44,49):
                    data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp/',str(class_num))
                    data_path = os.path.join(data_path,'img60_center_nonorm_per2_ep5k/models/1/eval_op','syn_data_%d.npy'%self.syn_per)                    
                    syn_data = np.load(data_path)
                    syn_samples = syn_samples+syn_data.shape[0]
                
                if self.only_syn == 1:
                    real_samples = 0
                else:
                    real_samples = self.data_3d.shape[0]
                real_syn_samples = syn_samples + real_samples

                # Initiate total data
                self.data_3d_tot = np.zeros((real_syn_samples,3,300,25,2),dtype=np.float32)
                self.num_frames_3d_tot = np.zeros((real_syn_samples),dtype=np.int32)
                self.video_name_tot = []
                self.labels_3d_tot = []
                self.type_label_tot = [] # 1-Real data ; 0-Synthetic data

                # Transfer real data
                if self.only_syn == 0:
                    self.data_3d_tot[0:real_samples,:,:,:,:] = self.data_3d
                    self.num_frames_3d_tot[0:real_samples] = self.num_frames_3d
                    
                    for i in range(real_samples):
                        self.video_name_tot.append(self.video_name[i])
                        self.labels_3d_tot.append(self.labels_3d[i])
                        self.type_label_tot.append(self.type_label[i])
                
                # Transfer synthetic data
                start_ind = real_samples
                for class_num in range(44,49):
                    data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp/',str(class_num))
                    data_path = os.path.join(data_path,'img60_center_nonorm_per2_ep5k/models/1/eval_op','syn_data_%d.npy'%self.syn_per)                    
                    syn_data = np.load(data_path)

                    num_samples = syn_data.shape[0]
                    end_ind = start_ind + num_samples

                    self.data_3d_tot[start_ind:end_ind,:,0:self.temporal_length,:,:] = syn_data
                    
                    for ind in range(start_ind,end_ind):
                        self.num_frames_3d_tot[ind] = self.temporal_length
                        self.video_name_tot.append("Synthetic.skeleton")
                        self.labels_3d_tot.append(class_num)
                        self.type_label_tot.append(0)

                    start_ind = end_ind
            
                self.data_3d = self.data_3d_tot
                self.video_name = self.video_name_tot
                self.num_frames_3d = self.num_frames_3d_tot
                self.labels_3d = self.labels_3d_tot
                self.type_label = self.type_label_tot

            #print (self.data_3d_inter.shape[0],self.num_frames_3d_inter.shape[0])
            #print (self.data_3d.shape[0],self.num_frames_3d.shape[0],len(self.video_name))

        print ("Loaded the %s set"%set_,"Total samples",len(self.video_name))   

    def __len__(self):
        if (self.expt == 'check'):
            return (len(self.video_name[0:10]))
        else:
            return (len(self.video_name))
    
    def __getitem__(self,id):

        # Get the labels
        self.labels = self.labels_3d[id]
        self.video = self.video_name[id]
        self.num_frames = self.num_frames_3d[id]
        self.type_ = self.type_label[id]
        if self.dataset != 'vacnn':
            self.video = self.video.replace(".skeleton","")

        # Get the data
        self.data = self.data_3d[id]
        self.data = np.nan_to_num(self.data)
        self.data[self.data == -inf] = 0

        # Get the deid data & convert nans to 0.0
        if self.temporal_pattern == 'interpolate':
            if self.type_ == 1:
                if self.centering == 1:
                    origin_data = self.data[:,:,1,0]
                    self.data = self.data - origin_data[:,:,None,None]
                if self.set_ == 'train':
                    p_interval = [0.5,1]
                    p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
                elif self.set_ == 'test':
                    p_interval = [0.95]
                    p = p_interval[0]
                #print (self.data.shape,self.num_frames)
                self.data = valid_crop_resize_multi_data(self.data,self.num_frames,p_interval,p,self.temporal_length)
                #self.data = self.data[:,:,:,0:1]
                #self.data = np.squeeze(self.data)
                if self.normalize == 1:
                    #max_val,min_val = 5.18858098984,-5.28981208801
                    max_val,min_val = 5.826573,-4.9881773
                    self.data = normalize_data(self.data,max_val,min_val,-1)
                if self.spherical == 1:
                    num_joints = self.data.shape[2]
                    data_sph = np.zeros((3,self.temporal_length,num_joints,2),dtype=np.float32)
                    for i in range(self.temporal_length):
                        for j in range(num_joints):
                            x,y,z = self.data[0,i,j,0], self.data[1,i,j,0], self.data[2,i,j,0]
                            #print (x,y,z)
                            az, el, r = cart2sph(x,y,z)
                            #print (az,el,r)
                            #print (x,y,z)
                            data_sph[0,:,:,0] = az
                            data_sph[1,:,:,0] = el
                            data_sph[1,:,:,0] = r
                    if self.centering == 1:
                        origin_data = data_sph[:,:,1,0]
                        data_sph = data_sph - origin_data[:,:,None,None]
                    self.data = data_sph
                #print (self.data.shape)
                #self.data = reshape_data_vacnn(self.data)
                #rgb_ske = convert_to_rgb_interpolate(self.data,max_val,min_val,150,0)
                #print (rgb_ske.shape)
            elif self.type_ == 0:
                self.data = self.data[:,0:self.temporal_length,:,:]

            return self.data,self.labels
        else:
            max_val,min_val = 5.18858098984,-5.28981208801
            self.data = reshape_data_vacnn(self.data)
            rgb_ske = convert_to_rgb_vacnn(self.data,max_val,min_val,150) # C*V*M = 150
            return rgb_ske,self.labels

class NTURGBDData1_classnum(Dataset):

    def __init__(self,temporal_length,temporal_pattern,gpu_id,dataset,split,normalize_flag,centering_flag,spherical_flag,class_num,set_,transform,expt):
        self.temporal_length = temporal_length
        self.labels = None
        self.temporal_pattern = temporal_pattern
        self.set_=set_
        self.split = split
        self.gpu_id = gpu_id
        self.dataset = dataset
        self.expt = expt
        self.normalize = normalize_flag
        self.centering = centering_flag
        self.spherical = spherical_flag
        self.class_num = class_num
        self.transform = transform

        if self.dataset == 'vacnn':
            self.data_path_3d = "/fast-stripe/datasets/nyu_action/ntu_3d_data_vacnn/"
            self.deid_path = "/fast-stripe/datasets/nyu_action/ntu_3d_data_vacnn/"
        elif self.dataset == 'dgnn':
            self.data_path_3d = "/fast-stripe/datasets/nyu_action/ntu_3d_data_dgnn/"
            self.deid_path = "/fast-stripe/datasets/nyu_action/deid_multiperson_data/"
        else:
            self.data_path_3d = "/fast-stripe/datasets/nyu_action/ntu_3d_data/"
            self.deid_path = "/fast-stripe/datasets/nyu_action/deid_multiperson_data/"

        data_3d = set_+"_"+split+"_"+"3ddata.npy"
        labels_3d = set_+"_"+split+"_"+"label_3ddata.pkl"
        num_frames_3d = set_+"_"+split+"_"+"num_frame_3ddata.npy"
        data_bone_3d = set_+"_"+split+"_"+"data_bone.npy"
        self.data_bone_3d_path = os.path.join(self.data_path_3d,data_bone_3d)
        self.data_3d_path = os.path.join(self.data_path_3d,data_3d)
        self.labels_3d_path = os.path.join(self.data_path_3d,labels_3d)
        self.num_frames_3d_path = os.path.join(self.data_path_3d,num_frames_3d)

        with open(self.labels_3d_path,"rb") as f:
            self.video_name,self.labels_3d = pickle.load(f)
        f.close()

        self.data_3d = np.load(self.data_3d_path)
        self.num_frames_3d = np.load(self.num_frames_3d_path)

        #============================================ Single-person ACTS ================================#
        ignore_samples_path = '/fast-stripe/datasets/nyu_action/samples_with_missing_skeleton.txt'
        with open(ignore_samples_path, 'r') as f:
            ignored_samples = [line.strip() for line in f.readlines()]
        f.close()

        tot_samples = 0
        for i in range(len(self.labels_3d)):
            if (self.labels_3d[i] == self.class_num): # <=48 : Single person acts ; 42 - Fall ; 26 - Jump
                video_name = self.video_name[i].replace(".skeleton","")
                if video_name not in ignored_samples:
                    tot_samples = tot_samples+1

        #print ("Total single activity samples",tot_samples)
        self.data_3d_inter = np.zeros((tot_samples,3,300,25,2),dtype=np.float32)
        self.num_frames_3d_inter = np.zeros((tot_samples),dtype=np.int32)
        self.video_name_inter = []
        self.labels_3d_inter = []
        inter_ind = 0
        sample_indices = []
        for i in range(len(self.labels_3d)):
            #print (self.labels_3d[i])
            if (self.labels_3d[i] == self.class_num):
                video_name = self.video_name[i].replace(".skeleton","")
                if video_name not in ignored_samples:
                    self.data_3d_inter[inter_ind] = self.data_3d[i]
                    #print (self.data_3d_inter[inter_ind])
                    self.num_frames_3d_inter[inter_ind] = self.num_frames_3d[i]
                    #print (self.num_frames_3d_inter)
                    self.video_name_inter.append(self.video_name[i])
                    self.labels_3d_inter.append(self.labels_3d[i])
                    sample_indices.append(i)
                    inter_ind = inter_ind + 1

        self.data_3d = self.data_3d_inter
        self.num_frames_3d = self.num_frames_3d_inter
        self.video_name = self.video_name_inter
        self.labels_3d = self.labels_3d_inter

        print ("Loaded the %s set"%set_,"Total samples",len(self.video_name))

    def __len__(self):
        if (self.expt == 'check'):
            return (len(self.video_name[0:10]))
        else:
            return (len(self.video_name))

    def __getitem__(self,id):

        # Get the labels
        self.labels = self.labels_3d[id]
        self.video = self.video_name[id]
        self.num_frames = self.num_frames_3d[id]
        if self.dataset != 'vacnn':
            self.video = self.video.replace(".skeleton","")

        # Get the data
        self.data = self.data_3d[id]
        self.data = np.nan_to_num(self.data)

        # Get the deid data & convert nans to 0.0
        if self.temporal_pattern == 'interpolate':
            if self.centering == 1:
                origin_data = self.data[:,:,1,0]
                self.data = self.data - origin_data[:,:,None,None]
            if self.set_ == 'train':
                p_interval = [0.5,1]
                p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
            elif self.set_ == 'test':
                p_interval = [0.95]
                p = p_interval[0]
            #print (self.data.shape,self.num_frames)
            self.data = valid_crop_resize_multi_data(self.data,self.num_frames,p_interval,p,self.temporal_length)
            #self.data = self.data[:,:,:,0:1]
            #self.data = np.squeeze(self.data)
            if self.normalize == 1:
                #max_val,min_val = 5.18858098984,-5.28981208801
                max_val,min_val = 5.826573,-4.9881773
                self.data = normalize_data(self.data,max_val,min_val,-1)
            if self.spherical == 1:
                num_joints = self.data.shape[2]
                data_sph = np.zeros((3,self.temporal_length,num_joints,2),dtype=np.float32)
                for i in range(self.temporal_length):
                    for j in range(num_joints):
                        x,y,z = self.data[0,i,j,0], self.data[1,i,j,0], self.data[2,i,j,0]
                        #print (x,y,z)
                        az, el, r = cart2sph(x,y,z)
                        #print (az,el,r)
                        #print (x,y,z)
                        data_sph[0,:,:,0] = az
                        data_sph[1,:,:,0] = el
                        data_sph[1,:,:,0] = r
                if self.centering == 1:
                    origin_data = data_sph[:,:,1,0]
                    data_sph = data_sph - origin_data[:,:,None,None]
                self.data = data_sph

            return self.data,self.labels
        else:
            max_val,min_val = 5.18858098984,-5.28981208801
            self.data = reshape_data_vacnn(self.data)
            rgb_ske = convert_to_rgb_vacnn(self.data,max_val,min_val,150) # C*V*M = 150
            return rgb_ske,self.labels

## FID score calculation for real dataset
class NTURGBDData1_real_fid_classnum(Dataset):

    def __init__(self,temporal_length,temporal_pattern,gpu_id,dataset,split,normalize_flag,centering_flag,spherical_flag,class_num,set_,transform,expt):
        self.temporal_length = temporal_length
        self.labels = None
        self.temporal_pattern = temporal_pattern
        self.set_=set_
        self.split = split
        self.gpu_id = gpu_id
        self.dataset = dataset
        self.expt = expt
        self.normalize = normalize_flag
        self.centering = centering_flag
        self.spherical = spherical_flag
        self.class_num = class_num
        self.transform = transform

        print ("Loading ",self.dataset, "data")
        if self.dataset == 'vacnn':
            self.data_path_3d = "/fast-stripe/datasets/nyu_action/ntu_3d_data_vacnn/"
            self.deid_path = "/fast-stripe/datasets/nyu_action/ntu_3d_data_vacnn/"
        elif self.dataset == 'dgnn':
            self.data_path_3d = "/fast-stripe/datasets/nyu_action/ntu_3d_data_dgnn/"
            self.deid_path = "/fast-stripe/datasets/nyu_action/deid_multiperson_data/"
        else:
            self.data_path_3d = "/fast-stripe/datasets/nyu_action/ntu_3d_data/"
            self.deid_path = "/fast-stripe/datasets/nyu_action/deid_multiperson_data/"

        data_3d = set_+"_"+split+"_"+"3ddata.npy"
        labels_3d = set_+"_"+split+"_"+"label_3ddata.pkl"
        num_frames_3d = set_+"_"+split+"_"+"num_frame_3ddata.npy"
        data_bone_3d = set_+"_"+split+"_"+"data_bone.npy"
        self.data_bone_3d_path = os.path.join(self.data_path_3d,data_bone_3d)
        self.data_3d_path = os.path.join(self.data_path_3d,data_3d)
        self.labels_3d_path = os.path.join(self.data_path_3d,labels_3d)
        self.num_frames_3d_path = os.path.join(self.data_path_3d,num_frames_3d)

        with open(self.labels_3d_path,"rb") as f:
            self.video_name,self.labels_3d = pickle.load(f)
        f.close()

        if self.dataset == 'vacnn':
            self.video_name = np.array(self.video_name)
            self.video_name = self.video_name.astype('U25')

        self.data_3d = np.load(self.data_3d_path)
        self.num_frames_3d = np.load(self.num_frames_3d_path)

        #============================================ Single-person ACTS ================================#
        ignore_samples_path = '/fast-stripe/datasets/nyu_action/samples_with_missing_skeleton.txt'
        with open(ignore_samples_path, 'r') as f:
            ignored_samples = [line.strip() for line in f.readlines()]
        f.close()

        tot_samples = 0
        for i in range(len(self.labels_3d)):
            if (self.labels_3d[i] == self.class_num): # <=48 : Single person acts ; 42 - Fall ; 26 - Jump
                if self.dataset != 'vacnn':
                    video_name = self.video_name[i].replace(".skeleton","")
                else:
                    video_name = self.video_name[i]
                if video_name not in ignored_samples:
                    tot_samples = tot_samples+1

        #print ("Total single activity samples",tot_samples)
        self.data_3d_inter = np.zeros((tot_samples,3,300,25,2),dtype=np.float32)
        self.num_frames_3d_inter = np.zeros((tot_samples),dtype=np.int32)
        self.video_name_inter = []
        self.labels_3d_inter = []
        inter_ind = 0
        sample_indices = []
        for i in range(len(self.labels_3d)):
            #print (self.labels_3d[i])
            if (self.labels_3d[i] == self.class_num):
                if self.dataset != 'vacnn':
                    video_name = self.video_name[i].replace(".skeleton","")
                else:
                    video_name = self.video_name[i]
                if video_name not in ignored_samples:
                    self.data_3d_inter[inter_ind] = self.data_3d[i]
                    #print (self.data_3d_inter[inter_ind])
                    self.num_frames_3d_inter[inter_ind] = self.num_frames_3d[i]
                    #print (self.num_frames_3d_inter)
                    self.video_name_inter.append(self.video_name[i])
                    self.labels_3d_inter.append(self.labels_3d[i])
                    sample_indices.append(i)
                    inter_ind = inter_ind + 1

        self.data_3d = self.data_3d_inter
        self.num_frames_3d = self.num_frames_3d_inter
        self.video_name = self.video_name_inter
        self.labels_3d = self.labels_3d_inter

        print ("Loaded the %s set"%set_,"Total samples",len(self.video_name))

    def __len__(self):
        if (self.expt == 'check'):
            return (len(self.video_name[0:10]))
        else:
            return (len(self.video_name))

    def __getitem__(self,id):

        # Get the labels
        self.labels = self.labels_3d[id]
        self.video = self.video_name[id]
        self.num_frames = self.num_frames_3d[id]
        if self.dataset != 'vacnn':
            self.video = self.video.replace(".skeleton","")

        # Get the data
        self.data = self.data_3d[id]
        self.data = np.nan_to_num(self.data)
        if self.dataset == 'vacnn':
            self.num_frames = calc_num_frames_vacnn(self.data,150)

        # Get the deid data & convert nans to 0.0
        if self.centering == 1:
            #print ("Centering the data")
            origin_data = self.data[:,:,1,0]
            self.data = self.data - origin_data[:,:,None,None]
        if self.set_ == 'train':
            p_interval = [0.5,1]
            p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
        elif self.set_ == 'test':
            p_interval = [0.95]
            p = p_interval[0]
        #print (self.data.shape,self.num_frames)
        self.data = valid_crop_resize_multi_data(self.data,self.num_frames,p_interval,p,self.temporal_length)
        #print (self.data.shape)
        #self.data = self.data[:,:,:,0:1]
        #self.data = np.squeeze(self.data)
        self.data[:,:,:,1:2] = np.zeros((3,self.temporal_length,25,1),dtype=np.float32)
        if self.temporal_pattern == 'interpolate':
            return self.data,self.labels
        else:
            max_val,min_val = 5.18858098984,-5.28981208801
            self.data = reshape_data_vacnn(self.data)
            rgb_ske = convert_to_rgb_vacnn(self.data,max_val,min_val,150) # C*V*M = 150
            return rgb_ske,self.labels

## FID score calculation for fake dataset
class NTURGBDDatasyn_classnum(Dataset):

    def __init__(self,temporal_length,temporal_pattern,gpu_id,gan,split,normalize_flag,centering_flag,spherical_flag,class_num,set_,transform,dims,expt):
        self.temporal_length = temporal_length
        self.labels = None
        self.temporal_pattern = temporal_pattern
        self.set_=set_
        self.split = split
        self.gpu_id = gpu_id
        self.gan = gan
        self.expt = expt
        self.normalize = normalize_flag
        self.centering = centering_flag
        self.spherical = spherical_flag
        self.class_num = class_num
        self.transform = transform
        self.dims = dims

        syn_samples = 0
        if self.gan == 'cgan':
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_center_nonorm_per1_ep5k/models/5000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_per1_ep500/models/500/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_bceloss_sigmoid_per1_ep5k/models/5000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_changelabels_per1_ep5k/models/1000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_largemodel_changelabels_per1_5p5k/models/5000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_mlpsixlayer_changelabels_per1_p5k/models/1000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan/img25_latentdim1024_changelabels_per1_ep5k/models/500/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan/realper10_z512_hcndata_cnn_sigmoid_mseloss_linearblock_per1_ep500/models/500/eval_op',str(class_num))
            data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan/realper10_z512_center_mlp_mseloss_per1_ep5k/models/1000/eval_op',str(class_num))
        elif self.gan == 'cgan_multi':
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_lambdacontr_0p1_featmapsnew_per1_ep5k/models/5000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_lambdatriplet_0p1_featmapsnew_per1_ep5k/models/5000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_cosineembed_0p5_featmapsnew_per1_ep5k/models/1000/eval_op',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_torchtriplet_5p0_featmapsnew_per1_ep5k/models/4000/eval_op',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_torchtriplet_0p1_featmapsnew_per1_ep5k/models/4000/eval_op',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_torchtriplet_1p0_featmapsnew_per1_ep5k/models/4000/eval_op',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_lambdacontr1p0_requiregrad_per1_ep5k/models/5000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_lambdatrip1p0_detach_per1_ep5k/models/5000/eval_op',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_lambdatrip1p0_updatefeatnw_per1_ep5k/models/500/eval_op',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/CCC_models/cgan_action_class/img25_lambdatrip5p0_margin2p0/models/2500/eval_op',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_nonorm_lambdatrip1p0_updatefeatnw_per1_ep5k/models/4000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/CCC_models/cgan_action_class/img25_lambdatriporg1p0_margin2p0/models/4000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_center_lambdacosine1p0_margin2p0_per1_ep5k/models/5000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_hcnmodeleval_lambdatriplet_1p0_per1_ep5k/models/500/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_concatfeatmap_per1_ep5k/models/5000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_hcnmodeleval_lambdatriplet_1p0_margin5p0_per1_ep5k/models/5000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_concatfeatmap_bcewithlogits_per1_ep5k/models/5000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_earlyconcat512_per1_ep5k/models/1000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_concatfeatmap256_per1_ep5k/models/5000/eval_op/',str(class_num))
            data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/cgan_action_class/img25_earlyconcatadd512_per1_ep5k/models/1000/eval_op/',str(class_num))
        elif self.gan == 'wgan_gp':
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp/CCC_models/',str(class_num),'img60_center_nonorm_ep5k/models/5000/eval_op/')
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp/',str(class_num),'img25_center_nonorm_per1_ep1k/models/1000/eval_op')
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp/',str(class_num),'img25_rerun/models/5000/eval_op')
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp/',str(class_num),'vacnn_img25_per1_ep5k/models/5000/eval_op')
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp/',str(class_num),'vacnn_calcframes_img25_per1_ep5k/models/5000/eval_op')
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp_action_class/',str(class_num),'img25_lambdatriplet_0p2_per1_ep5k/models/500/eval_op')
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp_action_class/',str(class_num),'img25_lambdatriplet_5p0_per1_ep5k/models/5000/eval_op')
            data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp/',str(class_num),'img25_scheduler_rerun/models/100/eval_op/')
        elif self.gan == 'wgan_gp_multi':
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp_multiclass/cgan/img25_multiclass_center_nonorm_per1/models/1000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp_multiclass/cgan/img25_center_change_model/models/1000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp_multiclass/cgan/img25_center_change_model_wgan_lossfn/models/5000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp_multiclass/cgan/vacnn_img25_change_model/models/5000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp_multiclass/cgan/vacnn_img25_center_change_model_changelossfnc/models/5000/eval_op/',str(class_num))
            #data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp_multiclass/cgan/img25_center_changelabels_changemodel/models/5000/eval_op/',str(class_num))
            data_path = os.path.join('/fast-stripe/workspaces/deval/synthetic-data/wgan_gp_multiclass/cgan/img25_center_changelabels_per1_ep5k/models/5000/eval_op/',str(class_num))

        if self.dims == 2048:
            data_path = os.path.join(data_path,'syn_2048_samples.npy')
        elif self.dims == 768:
            data_path = os.path.join(data_path,'syn_768_samples.npy')
        elif self.dims == 192:
            data_path = os.path.join(data_path,'syn_50000_samples.npy')              
        syn_data = np.load(data_path)
        syn_samples = syn_samples+syn_data.shape[0]
        
        self.num_frames_3d = np.zeros((syn_samples),dtype=np.int32)
        self.video_name = []
        self.labels_3d = []

        for i in range(syn_samples):
            self.num_frames_3d[i] = self.temporal_length
            self.video_name.append('Synthetic.skeleton')
            self.labels_3d.append(self.class_num)

        self.data_3d = syn_data
        
        print ("Loading samples from",data_path)
        print ("Loaded the %s set"%set_,"Total samples",len(self.video_name))

    def __len__(self):
        if (self.expt == 'check'):
            return (len(self.video_name[0:10]))
        else:
            return (len(self.video_name))

    def __getitem__(self,id):

        # Get the labels
        self.labels = self.labels_3d[id]
        self.video = self.video_name[id]
        self.num_frames = self.num_frames_3d[id]

        # Get the data
        self.data = self.data_3d[id]
        self.data = np.nan_to_num(self.data)

        # Get the deid data & convert nans to 0.0
        self.data = self.data[:,0:self.temporal_length,:,:]
        max_val,min_val = 5.18858098984,-5.28981208801
        self.data = reshape_data_vacnn(self.data)
        rgb_ske = convert_to_rgb_vacnn(self.data,max_val,min_val,150) # C*V*M = 150
        return rgb_ske,self.labels    