import os
import sklearn
import matplotlib
import numpy as np
import argparse

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from matplotlib import pyplot
import matplotlib.cm as cm

parser = argparse.ArgumentParser()
parser.add_argument("--t")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
opt = parser.parse_args()
print(opt)

data_path = '/fast-stripe/workspaces/deval/synthetic-data/hcn/feat_map/'
data_path_syn = '/fast-stripe/workspaces/deval/synthetic-data/hcn/feat_map/synthetic/cgan/'

classes = [i for i in range(0,49)]
classes_syn = []
#colors = iter(cm.rainbow(np.linspace(0, 1, len(classes)+len(classes_syn))))
colors = iter(cm.nipy_spectral(np.linspace(0, 1, len(classes)+len(classes_syn))))

all_feat_maps = np.zeros(((len(classes)+len(classes_syn))*32,512),dtype=np.float32)

ind = 0

for class_num in classes:

    feat_map_path = os.path.join(data_path,str(class_num),'feat_map_32.npy')
    feat_map = np.load(feat_map_path)

    all_feat_maps[ind*32:(ind+1)*32] = feat_map
    ind = ind + 1

if len(classes_syn) >= 1:
    for class_num in classes_syn:

        feat_map_path = os.path.join(data_path_syn,str(class_num),'feat_map_32.npy')
        feat_map = np.load(feat_map_path)

        all_feat_maps[ind*32:(ind+1)*32] = feat_map
        ind = ind + 1

# Visualize the feat_maps with TSNE
print ("Starting TSNE")
tsne = TSNE(n_components=2,verbose=1,random_state=42)
feat_maps_tsne = tsne.fit_transform(all_feat_maps)
#print (feat_maps_tsne)
print ("TSNE Completed")

# Plot the feature maps

fig, ax1 = pyplot.subplots(1,1)

for ind in range(len(classes)+len(classes_syn)):
    if len(classes) < 6:
        if ind < len(classes):
            ax1.scatter(feat_maps_tsne[ind*32:(ind+1)*32,0],feat_maps_tsne[ind*32:(ind+1)*32,1],marker='o',color=next(colors),label='R %d'%classes[ind])
        else:
            ax1.scatter(feat_maps_tsne[ind*32:(ind+1)*32,0],feat_maps_tsne[ind*32:(ind+1)*32,1],marker='o',color=next(colors),label='S %d'%classes_syn[ind-len(classes)])
    else:
        if ind < len(classes):
            ax1.scatter(feat_maps_tsne[ind*32:(ind+1)*32,0],feat_maps_tsne[ind*32:(ind+1)*32,1],marker='o',color=next(colors))
        else:
            ax1.scatter(feat_maps_tsne[ind*32:(ind+1)*32,0],feat_maps_tsne[ind*32:(ind+1)*32,1],marker='o',color=next(colors))
pyplot.legend(loc="upper right")
pyplot.show()