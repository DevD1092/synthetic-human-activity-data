import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from net.utils.tgcn import *
from net.utils.graph import Graph
from utils.common import *

def conv_init(module):
    # he_normal
    n = module.out_channels
    for k in module.kernel_size:
        n *= k
    module.weight.data.normal_(0, math.sqrt(2. / n))

class Encoder(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, n_z, graph_args,
                 edge_importance_weighting=False, temporal_kernel_size=59, **kwargs):
        super(Encoder,self).__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        #print ("Encoder in channels",in_channels)
        #print ("A",A.size())
        # build networks
        spatial_kernel_size = A.size(0)
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.encoder = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, **kwargs),
             st_gcn(64, 64, kernel_size, 1, **kwargs),
             st_gcn(64, 64, kernel_size, 1, **kwargs),
             #st_gcn(64, 64, kernel_size, 1, **kwargs),
             #st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 32, kernel_size, 1, **kwargs),
             st_gcn(32, 32, kernel_size, 1, **kwargs),
             #st_gcn(32, 32, kernel_size, 1, **kwargs),
             #st_gcn(32, 32, kernel_size, 1, **kwargs),
            st_gcn(32, 32, kernel_size, 1, **kwargs)
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.encoder
            ])
        else:
            self.edge_importance = [1] * len(self.encoder)

        # fcn for encoding
        #self.z_mean = nn.Conv2d(32, n_z, kernel_size=1)
        #self.z_lsig = nn.Conv2d(32, n_z, kernel_size=1)
        #self.fc1= nn.Sequential(
        #    nn.Linear(32,256*2), # out_channel *2 for temporal length 30 and out_channel *4 for temporal length 60
        #    nn.ReLU(),
        #    nn.Dropout2d(p=0.5))
        #self.fc2 = nn.Linear(256*2,1)
        self.fcn = nn.Conv1d(32, 1, kernel_size=1)
        conv_init(self.fcn)
        self.sig = nn.Sigmoid()

    def forward(self, x, l):

        # concat
        x = torch.cat((x, l), dim=1)

        # data normalization
        N, C, T, V, M = x.size()
        #print (N,C,T,V,M)
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        #print (x.shape)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.encoder, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        #print ("X size before pooling", x.size())
        
        ## STGCN Pooling
        # V pooling
        x = F.avg_pool2d(x, kernel_size=(1, V))
        #print('avg_pool2d V pooling',x.size())
        # M pooling
        x = x.view(N, M, x.size(1), x.size(2))
        #print('view M pooling',x.size())
        x = x.mean(dim=1)
        #print('mean M pooling',x.size())
        # T pooling
        x = F.avg_pool1d(x, kernel_size=x.size()[2])
        #print('avg_pool1d T pooling',x.size())


        ## Original pooling
        #x = F.avg_pool2d(x, x.size()[2:])
        #x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        #mean = self.z_mean(x)
        #mean = mean.view(mean.size(0), -1)
        #lsig = self.z_lsig(x)
        #lsig = lsig.view(lsig.size(0), -1)
        #print ("X before reshape",x.shape)
        #x = x.view(x.size(0), -1)
        #print ("X before fc",x.shape)
        #x = self.fc1(x)
        #x = self.fc2(x)

        x = self.fcn(x)
        #print('fcn', x.size())
        x = F.avg_pool1d(x, x.size()[2:])
        #print('x', x.size())
        x = x.view(N, 1)
        x = self.sig(x)

        #return mean, lsig
        return x


class Decoder(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, n_z, graph_args,
                 edge_importance_weighting=False, temporal_kernel_size=59, **kwargs):
        super(Decoder,self).__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.fcn = nn.ConvTranspose2d(n_z, 32, kernel_size=1)

        self.decoder = nn.ModuleList((
            st_gctn(32, 32, kernel_size, 1, **kwargs),
             st_gctn(32, 32, kernel_size, 1, **kwargs),
             #st_gctn(32, 32, kernel_size, 1, **kwargs),
             #st_gctn(32, 32, kernel_size, 1, **kwargs),
            st_gctn(32, 64, kernel_size, 1, **kwargs),
             #st_gctn(64, 64, kernel_size, 1, **kwargs),
             #st_gctn(64, 64, kernel_size, 1, **kwargs),
             st_gctn(64, 64, kernel_size, 1, **kwargs),
             st_gctn(64, 64, kernel_size, 1, **kwargs),
            st_gctn(64, in_channels, kernel_size, 1, ** kwargs)
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.decoder
            ])
        else:
            self.edge_importance = [1] * len(self.decoder)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        #self.out = nn.Sigmoid()

    def forward(self, z, l, T, V):

        N = z.size()[0]
        # concat
        z = torch.cat((z, l), dim=1)

        # reshape
        z = z.view(N, z.size()[1], 1, 1)

        # forward
        z = self.fcn(z)
        z = z.repeat([1, 1, T, V])
        # x = z.permute(0, 4, 3, 1, 2).contiguous()
        # x = x.view(N * M, V * C, T)
        #
        # x = self.data_bn(x)
        # x = x.view(N, M, V, C, T)
        # x = x.permute(0, 1, 3, 4, 2).contiguous()
        # x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.decoder, self.edge_importance):
            z, _ = gcn(z, self.A * importance)
        z = torch.unsqueeze(z, 4)

        # data normalization
        N, C, T, V, M = z.size()
        z = z.permute(0, 4, 3, 1, 2).contiguous()
        z = z.view(N * M, V * C, T)
        z = self.data_bn(z)
        z = z.view(N, M, V, C, T)
        z = z.permute(0, 3, 4, 2, 1).contiguous()
        # z = self.out(z)

        return z


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


class st_gctn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gctn = ConvTransposeTemporalGraphical(in_channels, out_channels,
                                                   kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        #print ("X", x.shape)
        #print ("A", A.shape)
        res = self.residual(x)
        #print ("Residual x",x.shape)
        x, A = self.gctn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A