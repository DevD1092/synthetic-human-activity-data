import numpy as np
import scipy.misc
import sys

def convert_to_rgb_vacnn(data,max_val,min_val,cvm,center_flag):
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
    
    #print ("Original skel data",ske_joint)
    # Convert to RGB
    ske_joint =  255 * (ske_joint - min_val) / (max_val - min_val)
    #print ("Image converted ske joint",ske_joint)
    #print (ske_joint.shape)
    rgb_ske = np.reshape(ske_joint, (ske_joint.shape[0], ske_joint.shape[1] //3, 3))
    #print ("Image converted ske joint 3 channel",rgb_ske)
    if rgb_ske.shape[0]==0:
        rgb_ske = np.zeros([224,224,3],dtype=np.float32)
    else:
        rgb_ske = scipy.misc.imresize(rgb_ske, (224, 224)).astype(np.float32)
    #rgb_ske = np.array(Image.fromarray(rgb_ske.astype('uint8')).resize((224,224))).astype(np.float32)
    #print ("RGB SKE 224x224 image",rgb_ske)
    if center_flag:
        rgb_ske = center(rgb_ske)
    rgb_ske = np.transpose(rgb_ske, [1, 0, 2])
    rgb_ske = np.transpose(rgb_ske, [2,1,0])
    return rgb_ske

def convert_to_rgb_interpolate(data,max_val,min_val,cvm,center_flag):
    #print ("Converting to an image")
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
    
    #print ("Original skel data",ske_joint)
    # Convert to RGB
    ske_joint =  255 * (ske_joint - min_val) / (max_val - min_val)
    #print ("Image converted ske joint",ske_joint)
    #print (ske_joint.shape)
    rgb_ske = np.reshape(ske_joint, (ske_joint.shape[0], ske_joint.shape[1] //3, 3))
    #print ("Image converted ske joint 3 channel",rgb_ske)
    #print (rgb_ske.shape)
    rgb_ske = np.transpose(rgb_ske, [1, 0, 2])
    rgb_ske = np.transpose(rgb_ske, [2,1,0])
    return rgb_ske

def center(rgb):
    rgb[:,:,0] -= 110
    rgb[:,:,1] -= 110
    rgb[:,:,2] -= 110
    return rgb