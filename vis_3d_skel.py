import numpy as np
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import pandas as pd
import pickle
import cv2
import scipy.misc
import argparse
import subprocess

from mpl_toolkits.mplot3d import Axes3D
from os import walk
from PIL import Image
from rgb_to_3dske import convert_vacnn_rgb_toskes,convert_vacnn_rgb_toskes_interpolate
from data3d_to_rgb import convert_to_rgb_vacnn,convert_to_rgb_interpolate
from interpolate import valid_crop_resize_multi_data
from tifffile import imsave,imread
from savitzky_golay import savitzky_golay

global args

parser = argparse.ArgumentParser(description='Visualize 3d activity')
parser.add_argument('--plot_actv', default=0, type=int,
                    help='Plot the 3d activity')
parser.add_argument('--data3d_to_rgb', default=0, type=int,
                    help='Convert the 3d data to rgb')
parser.add_argument('--rgb_to_3d', default=0, type=int,
                    help='Convert the rgb to 3d data')
parser.add_argument('--interpolate', default=0, type=int,
                    help='interpolate or not')
parser.add_argument('--temporal_length', default=60, type=int,
                    help='Temporal length to interpolate')
parser.add_argument('--normalize',default=0,type=int,
                    help='Normalize the data or not')
parser.add_argument('--centering',default=0,type=int,
                    help='Joint center the data or not')
parser.add_argument('--spherical',default=0,type=int,
                    help='Spherical coords or not')
parser.add_argument('--dataset',default='orig',type=str,
                    help='which type of dataset')
parser.add_argument('--smoothing',default=0,type=int,
                    help='To smooth or not')
args = parser.parse_args()

if args.dataset == 'orig':
    skels_data_path = '/fast-stripe/datasets/nyu_action/ntu_3d_data/train_sub_3ddata.npy'
    labels_data_path = '/fast-stripe/datasets/nyu_action/ntu_3d_data/train_sub_label_3ddata.pkl'
    num_frames_path = '/fast-stripe/datasets/nyu_action/ntu_3d_data/train_sub_num_frame_3ddata.npy'
elif args.dataset == 'vacnn':
    skels_data_path = '/fast-stripe/datasets/nyu_action/ntu_3d_data_vacnn/train_sub_3ddata.npy'
    labels_data_path = '/fast-stripe/datasets/nyu_action/ntu_3d_data_vacnn/train_sub_label_3ddata.pkl'
    num_frames_path = '/fast-stripe/datasets/nyu_action/ntu_3d_data_vacnn/train_sub_num_frame_3ddata.npy'
data_index = 102 # (33,79-fall, 127,54 - jump) ntu_data ; (102,42 - fall) vacnn_data

#========================================================= START - Getting activity pattern of the class ======================================#
'''
with open(labels_data_path,"rb") as f:
    video_name,labels_3d = pickle.load(f)
f.close()

if args.dataset == 'vacnn':
    video_name = np.array(video_name)
    video_name = video_name.astype('U25')

data_3d = np.load(skels_data_path)
count=0

for ind,video in enumerate(video_name):
    if count == 2:
        sys.exit()
    ind_cl = video.find('A')
    act_class = video[ind_cl+1:ind_cl+4]
    print (int(act_class))
    if (int(act_class) == 43):
        data_index = ind
        print ("Video",video)
        print ("Data index",data_index)
        data_rgb = data_3d[data_index]
        np.save('ind'+str(data_index)+'.npy',data_rgb)
        count = count + 1'''
#========================================================= END - Getting activity pattern of the class ======================================#

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

def convert_to_rgb(data):
    #print (data.shape)
    data = np.transpose(data,(1,3,2,0))
    data = np.reshape(data,(data.shape[0],data.shape[1],data.shape[2]*data.shape[3]))
    data = np.reshape(data,(data.shape[0],data.shape[1]*data.shape[2]))
    rgb_ske = np.reshape(data, (data.shape[0], data.shape[1] //3, 3))
    rgb_ske = scipy.misc.imresize(rgb_ske, (224, 224)).astype(np.float32)
    rgb_ske = np.transpose(rgb_ske, [1, 0, 2])
    rgb_ske = np.transpose(rgb_ske, [2,1,0])
    return rgb_ske

def normalize_data(data,max_val,min_val,range_num):
    if range_num == 0:
        data = (data-min_val)/(max_val-min_val)
    else:
        data = 2*(data-min_val)/(max_val-min_val) - 1
    return data

def denormalize_data(data,max_val,min_val,range_num):
    if range_num == 0:
        data = data*(max_val-min_val) + min_val
    else:
        data = (data + 1)*(max_val-min_val)/2 + min_val
    return data

def remove_files(dirname):
    files = []
    for root,directory,filename in os.walk(dirname):
        files.extend(filename)
        break

    for filename in files:
        filepath=os.path.join(dirname,filename)
        os.remove(filepath)

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

def vis_3dske(data_3d,data_index,num_frames_3d,video_name,flag_rgb_to_3d,save_dir,normalize_flag,centering_flag):

    #if normalize_flag == 1:
    #    max_val,min_val = 5.18858098984,-5.28981208801
    #    data_3d = denormalize_data(data_3d,max_val,min_val,-1)

    data_per1 = data_3d[:,0:num_frames_3d,:,0:1]
    data_per2 = data_3d[:,0:num_frames_3d,:,1:2]
    #print (data_per1)
    video = video_name[data_index]
    print ("Video",video)
    if (not np.any(data_per2)):
        per2_flag = 0
    else:
        per2_flag = 1
    #print (per2_flag)

    if normalize_flag == 1 or centering_flag == 1:
        per2_flag = 0

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')

    for frame in range(num_frames_3d):
        print ("Frame",frame)

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

        if (per2_flag == 1):

            world_head_bones_per2_x = []
            world_head_bones_per2_y = []
            world_head_bones_per2_z = []

            world_head_per2 = data_per2[:,frame,:,:]
            world_head_per2 = np.squeeze(world_head_per2,axis=2)
            world_head_per2 = np.transpose(world_head_per2,(1,0))

            world_head_per2_x = world_head_per2[:,0]
            world_head_per2_y = world_head_per2[:,1]
            world_head_per2_z = world_head_per2[:,2]

            for i in range(len(joint_seq)):
                #print (world_head_x[joint_seq[i][0]],world_head_x[joint_seq[i][1]])
                world_head_bones_per2_x.append([world_head_per2_x[joint_seq[i][0]-1],world_head_per2_x[joint_seq[i][1]-1]])
                world_head_bones_per2_y.append([world_head_per2_y[joint_seq[i][0]-1],world_head_per2_y[joint_seq[i][1]-1]])
                world_head_bones_per2_z.append([world_head_per2_z[joint_seq[i][0]-1],world_head_per2_z[joint_seq[i][1]-1]])

            world_head_bones_per2_x = np.array(world_head_bones_per2_x)
            world_head_bones_per2_y = np.array(world_head_bones_per2_y)
            world_head_bones_per2_z = np.array(world_head_bones_per2_z)

        plt.cla()
        
        for i in range(len(joint_seq)):
            ax2.plot(world_head_bones_per1_z[i],world_head_bones_per1_x[i],world_head_bones_per1_y[i],color='blue')
            if (per2_flag == 1):
                ax2.plot(world_head_bones_per2_x[i],world_head_bones_per2_y[i],world_head_bones_per2_z[i],color='blue')
        
        ax2.scatter(world_head_per1_z,world_head_per1_x,world_head_per1_y,s=50,label='True Position')
        if (per2_flag == 1):
            ax2.scatter(world_head_per2_x,world_head_per2_y,world_head_per2_z,s=50,label='True Position')
        if normalize_flag == 1:
            ax2.set_xlim(0.5,1.0) #-0.5,0.5
            ax2.set_ylim(0.5,1.0) #-0.5,0.5
            ax2.set_zlim(0.5,1.0) #-0.5,0.5
        if centering_flag == 1:
            ax2.set_xlim(-1,4.0) #-0.5,0.5
            ax2.set_ylim(-1,2.0) #-0.5,0.5
            ax2.set_zlim(-1,2.0) #-0.5,0.5
        else:
            ax2.set_xlim(-1,4)
            ax2.set_ylim(-1,2)
            ax2.set_zlim(-1,2)
        #if flag_rgb_to_3d == 1:
        #    ax2.set_xlim(0,1)
        #    ax2.set_ylim(0,1)
        #    ax2.set_zlim(0,1)
        ax2.set_title('Bones, frame={}'.format(frame))        
        
        plt.show()
        fig2.show()
        plt.savefig(save_dir + "/im%03d.png" % frame)
        plt.pause(0.05)

with open(labels_data_path,"rb") as f:
    video_name,labels_3d = pickle.load(f)
f.close()

#data_3d = np.load(skels_data_path)
#data_rgb = data_3d[data_index]
data_rgb = np.load('ind'+str(data_index)+'.npy')
if args.centering == 1:
    origin_data = data_rgb[:,:,1,0] # Joint number 2 of the 1st person taken as the origin
    data_rgb = data_rgb - origin_data[:,:,None,None]
num_frames_3d = np.load(num_frames_path)
num_frames = num_frames_3d[data_index]
#print ("Number of frames",num_frames)
if args.dataset == 'vacnn':
    num_frames = calc_num_frames_vacnn(data_rgb,150)
print ("Number of frames",num_frames)
#num_joints = data_3d.shape[3]
num_joints = data_rgb.shape[2]
max_val,min_val = 5.18858098984,-5.28981208801
vacnn_center_flag = 0

joint_seq = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
        (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
        (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
        (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
        (22, 23), (23, 8), (24, 25), (25, 12)]

#========================================================= Smooth the 3d data ==========================================#
if args.smoothing == 1:
    
    window_size = 7
    order = 3

    print ("Savitzky Golay hyperaparameters --> Window size: %d ; Order: %d"%(window_size,order))
    data_rgb_smooth = np.zeros((3,300,25,2),dtype=np.float32)

    print (data_rgb.shape)
    for per in range(data_rgb.shape[3]):
        for joint in range(data_rgb.shape[2]):
            x_coord = data_rgb[0,:,joint,per]
            y_coord = data_rgb[1,:,joint,per]
            z_coord = data_rgb[2,:,joint,per]
            #print (x_coord)
            x_coord_smooth = savitzky_golay(x_coord, window_size, order, deriv = 0)
            #print (x_coord_smooth)
            y_coord_smooth = savitzky_golay(y_coord, window_size, order, deriv = 0)
            z_coord_smooth = savitzky_golay(z_coord, window_size, order, deriv = 0)

            data_rgb_smooth[0,:,joint,per] = x_coord_smooth
            data_rgb_smooth[1,:,joint,per] = y_coord_smooth
            data_rgb_smooth[2,:,joint,per] = z_coord_smooth
    
    data_rgb = data_rgb_smooth



#========================================================= Convert RGB to 3d data ======================================#

if args.rgb_to_3d == 1:
    #rgb_vacnn = Image.open('vacnn_img_10_int8.png')
    #rgb_vacnn = Image.open('vacnn_img_interpolate_uint8_'+str(data_index)+'.png')
    rgb_vacnn = imread('vacnn_img_interpolate_tiff_'+str(data_index)+'.tif')
    #print (rgb_vacnn)
    num_per = 1
    num_frames = args.temporal_length
    #data_rgb = convert_vacnn_rgb_toskes(rgb_vacnn,num_frames,150,num_per,num_joints,max_val,min_val)
    data_rgb = convert_vacnn_rgb_toskes_interpolate(rgb_vacnn,num_frames,150,num_per,num_joints,max_val,min_val)
    #print (data_rgb)
    #print (data_rgb.shape)

#========================================================= Interpolate activity ======================================#

if args.interpolate == 1:

    #print ("Before interpolate",data_rgb)
    p_interval = [0.5,1]
    p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
    data_rgb = valid_crop_resize_multi_data(data_rgb,num_frames,p_interval,p,args.temporal_length)
    num_frames = args.temporal_length
    if args.spherical == 1:
        data_sph = np.zeros((3,num_frames,num_joints,2),dtype=np.float32)
        for i in range(num_frames):
            for j in range(num_joints):
                x,y,z = data_rgb[0,i,j,0], data_rgb[1,i,j,0], data_rgb[2,i,j,0]
                #print (x,y,z)
                az, el, r = cart2sph(x,y,z)
                #print (az,el,r)
                x,y,z = sph2cart(az,el,r)
                #print (x,y,z)
                data_sph[0,:,:,0] = az
                data_sph[1,:,:0] = el
                data_sph[1,:,:0] = r
        data_rgb = data_sph
    if args.normalize == 1:
        data_rgb = normalize_data(data_rgb,max_val,min_val,0)
    else:
        if args.centering == 0:
            data_rgb_vacnn = reshape_data_vacnn(data_rgb)
            rgb_ske = convert_to_rgb_interpolate(data_rgb_vacnn,max_val,min_val,150,vacnn_center_flag)
            rgb_ske = np.transpose(rgb_ske,(2,1,0))
    #imsave('vacnn_img_interpolate_tiff_'+str(data_index)+'.tif',rgb_ske)
    #sys.exit()
    #img_vacnn = Image.fromarray(rgb_ske.astype('float32'), 'RGB')
    #img_vacnn = Image.fromarray(rgb_ske, 'RGB')
    #img_vacnn.show()
    #print ("Image before saving",np.array(img_vacnn))
    #img_vacnn.save('vacnn_img_interpolate_fl32'+str(data_index)+'.png')
    #print ("After interpolate",data_rgb)

#========================================================= Convert to RGB ======================================#
if args.data3d_to_rgb == 1:
    data_rgb_vacnn = reshape_data_vacnn(data_rgb)
    rgb_ske = convert_to_rgb_vacnn(data_rgb_vacnn,max_val,min_val,150,vacnn_center_flag)
    #print (rgb_ske)
    rgb_ske = np.transpose(rgb_ske,(2,1,0))
    #img_vacnn = Image.fromarray(rgb_ske.astype('uint8'), 'RGB')
    img_vacnn = Image.fromarray(rgb_ske, 'RGB')
    img_vacnn.show()
    img_vacnn.save('vacnn_img_'+str(data_index)+'.png')
    rgb_ske_normal = convert_to_rgb(data_rgb)
    #print (rgb_ske_normal)
    rgb_ske_normal = np.transpose(rgb_ske_normal,(2,1,0))
    #img_normal = Image.fromarray(rgb_ske_normal.astype('uint8'), 'RGB')
    img_normal = Image.fromarray(rgb_ske_normal, 'RGB')
    img_normal.show()
    img_normal.save('normal_img'+str(data_index)+'.png')

#========================================================= Visualize activity ======================================#

if args.plot_actv == 1:

    save_dir = "vis_act"
    if args.smoothing == 1:
        save_dir = "vis_act_smooth"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    remove_files(save_dir)
    
    vis_3dske(data_rgb,data_index,num_frames,video_name,args.rgb_to_3d,save_dir,args.normalize,args.centering)
    #sys.exit()
    # Create a video of the plot images
    os.chdir(save_dir)
    subprocess.call([
        'ffmpeg', '-framerate', '1', '-i', 'im%03d.png', '-r', '1', '-pix_fmt', 'yuv420p',
        'vis_act.mp4'])