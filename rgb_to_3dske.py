import sys
import numpy as np
import scipy.misc

def convert_vacnn_rgb_toskes(rgb_vacnn,num_frames,cvm,num_per,num_joints,max_val,min_val):

    print (np.array(rgb_vacnn))
    #print (np.array(rgb_vacnn).shape)
    rgb_ske = scipy.misc.imresize(np.array(rgb_vacnn),(num_frames,num_joints*num_per)).astype(np.float32)
    print (rgb_ske)
    #print (rgb_ske.shape)
    if num_per == 1:
        ske_joint = np.reshape(rgb_ske,(rgb_ske.shape[0],cvm//2))
    elif num_per == 2:
        ske_joint = np.reshape(rgb_ske,(rgb_ske.shape[0],cvm))
    #print (ske_joint)
    #print (ske_joint.shape)
    #ske_joint = ske_joint * (max_val - min_val) / 255 + min_val
    ske_joint = (ske_joint - np.min(ske_joint)) / (np.max(ske_joint) - np.min(ske_joint))
    #print (ske_joint)
    #print (ske_joint.shape)
    ske_joint = np.reshape(ske_joint,(ske_joint.shape[0],ske_joint.shape[1]//3,3))
    #print (ske_joint)
    #print (ske_joint.shape)
    if num_per == 1:
        ske_joint = np.transpose(ske_joint,(2,0,1))
        ske_joint = np.expand_dims(ske_joint,axis=3)
        per2 = np.zeros((3,num_frames,num_joints,1),dtype=np.float32)
        ske_joint = np.stack((ske_joint,per2),axis=3)
        ske_joint = np.squeeze(ske_joint,axis=4)
        #print (ske_joint)
        #print (ske_joint.shape)
    return ske_joint

def convert_vacnn_rgb_toskes_interpolate(rgb_vacnn,num_frames,cvm,num_per,num_joints,max_val,min_val):

    print (np.array(rgb_vacnn))
    print (np.array(rgb_vacnn).shape)
    rgb_ske = np.array(rgb_vacnn)
    rgb_ske = np.transpose(rgb_ske,(1,0,2))
    print (rgb_ske.shape)
    #rgb_ske = scipy.misc.imresize(np.array(rgb_vacnn),(num_frames,num_joints*num_per)).astype(np.float32)
    rgb_ske = np.reshape(rgb_ske,(num_frames,num_joints*num_per*3))
    print (rgb_ske)
    #print (rgb_ske.shape)
    if num_per == 1:
        ske_joint = np.reshape(rgb_ske,(rgb_ske.shape[0],cvm//2))
    elif num_per == 2:
        ske_joint = np.reshape(rgb_ske,(rgb_ske.shape[0],cvm))
    print (ske_joint)
    #print (ske_joint.shape)
    ske_joint = ske_joint * (max_val - min_val) / 255 + min_val
    #ske_joint = (ske_joint - np.min(ske_joint)) / (np.max(ske_joint) - np.min(ske_joint))
    print (ske_joint)
    #print (ske_joint.shape)
    ske_joint = np.reshape(ske_joint,(ske_joint.shape[0],ske_joint.shape[1]//3,3))
    print (ske_joint)
    #print (ske_joint.shape)
    if num_per == 1:
        ske_joint = np.transpose(ske_joint,(2,0,1))
        ske_joint = np.expand_dims(ske_joint,axis=3)
        per2 = np.zeros((3,num_frames,num_joints,1),dtype=np.float32)
        ske_joint = np.stack((ske_joint,per2),axis=3)
        ske_joint = np.squeeze(ske_joint,axis=4)
        #print (ske_joint)
        #print (ske_joint.shape)
    return ske_joint