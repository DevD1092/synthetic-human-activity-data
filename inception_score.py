import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3
from torch.utils.data import DataLoader

import numpy as np
from scipy.stats import entropy
from dataloader import NTURGBDData,NTURGBDData1,NTURGBDData_full,NTURGBDData1_classnum,NTURGBDDatasyn_classnum,NTURGBDData1_real_fid_classnum

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--centering",type=int,default=0,help="Center the data or not")
parser.add_argument("--splits",type=int,default=10,help="Center the data or not")
parser.add_argument("--normalize",type=int,default=0,help="Normalize the data or not")
parser.add_argument("--spherical",type=int,default=0,help="Spherical the data or not")
parser.add_argument('--expt',default='check',type=str,help='check or final')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument("--class_batch",type=int,default=-1,help="Class batch 0 to 11")
parser.add_argument("--save_stats",type=int,default=0,help="Save mu sigma of the feature maps")
parser.add_argument("--gan",type=str,default='cgan',help="Which GAN output to compare")
parser.add_argument("--dataset",type=str,default='hcn',help="Which dataset to use")
parser.add_argument('--temporal_length', type=int, default=60,
                    help='Temporal length')
parser.add_argument('--dims', type=int, default=2048,
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))


def inception_score(ntuloader, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(ntuloader)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = DataLoader(ntuloader,batch_size=32,shuffle=True,num_workers=8,pin_memory=True)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, (batch,_) in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':

    args = parser.parse_args()

    classes = [0, 1, 2, 3]

    for class_num in classes:

        print ("Class num: ",class_num)
        # Configure data loader
        transform = None
        #syndata = NTURGBDDatasyn_classnum(args.temporal_length,None,args.gpu,args.gan,'sub',args.normalize,args.centering,args.spherical,class_num,'train',transform,args.dims,args.expt)
        syndata = NTURGBDData1_real_fid_classnum(args.temporal_length,None,args.gpu,args.dataset,'sub',args.normalize,args.centering,args.spherical,class_num,'train',transform,args.expt)
        loader1 = syndata

        print ("Calculating Inception Score...")
        print (inception_score(loader1, cuda=True, batch_size=32, resize=True, splits=args.splits))