
import torch
import torch.nn as nn
import re
#---
class Fc_PDEpara(nn.Module):
    def __init__(self, in_feature, out_feature,  nDepth=1, ClassStyle = '', OutValueRange=0.2 ): #bNormalize=0,
        super(Fc_PDEpara, self).__init__()
        self.OutValueRange = OutValueRange
        # if bNormalize:
        #     self.fc_net = nn.Sequential(nn.Linear( in_feature,  15),   nn.LayerNorm(15),  nn.ReLU(),
        #                                 nn.Linear( 15, 15 ), nn.LayerNorm(15),  nn.ReLU(),
        #                                 nn.Linear( 15, out_feature ), nn.Sigmoid()                       )
        if nDepth != 1 and 'O' in ClassStyle: # 'O' stands for OneHot_input
            self.nDepth_OneHotInput = nDepth
            in_feature = in_feature + self.nDepth_OneHotInput
            out_feature = out_feature//self.nDepth_OneHotInput

        if 'm' in ClassStyle.casefold():  # MLP
            self.fc_net = nn.Sequential( nn.Linear( in_feature , 50 ), nn.ReLU(),
                                         nn.Linear( 50, out_feature ), nn.Sigmoid() )
        elif 'l' in ClassStyle.casefold(): # linear
            self.fc_net = nn.Sequential( nn.Linear( in_feature, out_feature ), nn.Sigmoid() )

        else: #  # deep net or default
            m = 50   if 'd50' in ClassStyle.casefold() else  15
            self.fc_net = nn.Sequential(nn.Linear( in_feature,   m),  nn.ReLU(),
                                        nn.Linear( m, m ),           nn.ReLU(),
                                        nn.Linear( m, out_feature ),  nn.Sigmoid()             )


    def forward( self, x):
        # x.shape = batch, num_params
        if hasattr(self,'nDepth_OneHotInput'):
            batchsize = x.shape[0]
            x = x.unsqueeze(1).repeat(1,self.nDepth_OneHotInput,1)
            onehot_input_depth = torch.nn.functional.one_hot( torch.arange(self.nDepth_OneHotInput) ).unsqueeze(0).repeat(batchsize,1,1).to(x.device)
            x = torch.cat( (x, onehot_input_depth ), dim = -1 )
            x = self.fc_net(x).view( batchsize, -1 )
        else:
            x = self.fc_net(x)

        if hasattr(self,'OutValueRange'):   return  self.OutValueRange + (1-2*self.OutValueRange)* x
        else:                               return  0.2                + 0.6* x



@staticmethod
def calc_activation_shape( dim, ksize=(3,3), dilation=(1, 1), stride=(1, 1), padding=(1, 1) ):
    def shape_each_dim(i):
        odim_i = dim[i] + 2 * padding[i] - dilation[i] * (ksize[i] - 1) - 1
        return (odim_i / stride[i]) + 1
    return shape_each_dim(0), shape_each_dim(1)




#------------------------------
#------------------------------------------------------------------------------
class MyConvNd(nn.Module):  # keep strid ==1
    def __init__(self, nDIM, in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular',
                 bias=True, bRelu=True, bNorm=False, type='Conv'):
        super(MyConvNd, self).__init__()
        self.nDIM = nDIM

        self.type = type
        if bNorm ==True:   bias = False  # when either 'batch_norm'  or layer_norm are on, bias becomes redundent

        if 'r' == type.casefold()[0]:  # e.g 'Residual' , 'Residual1', 'Resid3'
            numRepeat = int(type[-1]) if type[-1].isdigit() else 1  # the repeat time is given by the last character (digit) of the given type
            self.net = ResidualBlockNd(nDIM, numRepeat, in_channels, out_channels, kernel_size, stride, padding,
                                       padding_mode, bias, bRelu, bNorm)
        else:

            layers = []
            # ----------------------------------------------------------------------
            if 'c' in type.casefold()[0]: # standard CNN
                # default parameter setting for learning flame stability
                if kernel_size == 1:
                    padding = 0
                elif kernel_size == 3:
                    padding = 1
                    padding_mode = 'circular'
                layers.append(nn_ConvNd(nDIM)(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                              padding_mode=padding_mode, bias=bias))
            elif 't' == type.casefold()[0]:
                layers.append(
                    nn_ConvTransposeNd(nDIM)(in_channels, out_channels, kernel_size, stride=stride, bias=bias))
            elif 'i' in type.casefold()[0]:
                layers.append(InceptionND_v3(nDIM, in_channels, out_channels))
            else:
                raise ValueError(type + ' is not found: MyConv1d')
            # ----------------------------------------------------------------------


            if bNorm :   layers.append( my_NormNd(nDIM,out_channels,bNorm)  )

            if bRelu == True:     layers.append(nn.ReLU())

            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


###########################################################################
#  Copy from torch.nn.modules.padding (2023), can be deleted if torch 2023 is installed
#########################################################################
from torch.nn.modules.module import Module
from torch import Tensor
from torch.nn.modules.utils import _pair, _quadruple, _ntuple
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t, _size_4_t, _size_6_t
from typing import Sequence, Tuple

class _CircularPadNd(Module):
    __constants__ = ['padding']
    padding: Sequence[int]
    def _check_input_dim(self, input):
        raise NotImplementedError
    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        return F.pad(input, self.padding, 'circular')
    def extra_repr(self) -> str:
        return f'{self.padding}'

class torch_nn_CircularPad1d(_CircularPadNd):
    padding: Tuple[int, int]
    def __init__(self, padding: _size_2_t) -> None:
        super().__init__()
        self.padding = _pair(padding)

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(                f"expected 2D or 3D input (got {input.dim()}D input)"            )
class torch_nn_CircularPad2d(_CircularPadNd):
    padding: Tuple[int, int, int, int]
    def __init__(self, padding: _size_4_t) -> None:
        super().__init__()
        self.padding = _quadruple(padding)
    def _check_input_dim(self, input):
        if input.dim() != 3 and input.dim() != 4:
            raise ValueError(              f"expected 3D or 4D input (got {input.dim()}D input)"            )
#########################################################################
#
#########################################################################

def nn_CircularPadNd(nDIM): # RY add the 'circlar' pad, April 9, 2024
    if nDIM==1:
        return torch_nn_CircularPad1d
    elif nDIM==2:
        return torch_nn_CircularPad2d
    else:
        raise ValueError('nn_CircularPadNd: nDIM='+str(nDIM) )
#------------------------------

def nn_ConvNd(nDIM):
    if nDIM==1:
        return nn.Conv1d
    elif nDIM==2:
        return nn.Conv2d
    else:
        raise ValueError('nn_ConvNd: nDIM='+str(nDIM) )
def nn_ConvTransposeNd(nDIM):
    if nDIM==1:
        return nn.ConvTranspose1d
    elif nDIM==2:
        return nn.ConvTranspose2d
    else:
        raise ValueError('nn_ConvTransposeNd: nDIM='+str(nDIM) )
def nn_MaxPoolNd(nDIM):
    if nDIM==1:
        return nn.MaxPool1d
    elif nDIM==2:
        return nn.MaxPool2d
    else:
        raise ValueError('nn_MaxPoolNd: nDIM='+str(nDIM) )
def nn_AvgPoolNd(nDIM):
    if nDIM == 1:
        return nn.AvgPool1d
    elif nDIM == 2:
        return nn.AvgPool2d
    else:
        raise ValueError('nn_AvgPoolNd: nDIM=' + str(nDIM))

def my_NormNd(nDIM, out_channel, bNorm):
    if nDIM == 1:
        if bNorm == -1:   return nn.BatchNorm1d(out_channel)
        else:           #  elif bNorm>0:
            return nn.LayerNorm( [out_channel, bNorm] )
    elif nDIM == 2:
        if bNorm == -1:   return nn.BatchNorm2d(out_channel)
        else:
            return nn.LayerNorm( [out_channel, bNorm[0], bNorm[1] ] )

    raise ValueError('my_NormNd: nDIM={}, out_channel={}, bNorm={}', nDIM, out_channel,bNorm)
#
# def nn_BatchNormNd(nDIM):
#     if nDIM == 1:
#         return nn.BatchNorm1d
#     elif nDIM == 2:
#         return nn.BatchNorm2d
#     else:
#         raise ValueError('nn_BatchNormNd: nDIM=' + str(nDIM))
#

# -------------------
class InceptionND_v3(nn.Module):
    def __init__(self, nDIM, in_fts, out_fts):
        super(InceptionND_v3, self).__init__()
        self.nDIM = nDIM

        # nn_ConvNd = nn.Conv1d if nDIM==1 else nn.Conv2d
        if type(out_fts) is not list:
            out_fts = [out_fts // 4, out_fts // 4, out_fts // 4, out_fts // 4]
        ###################################
        ### 1x1 conv + 3x3  conv + 3x3 conv
        ###################################
        self.branch1 = nn.Sequential(
            nn_ConvNd(nDIM)(in_channels=in_fts, out_channels=out_fts[0], kernel_size=1, stride=1),
            nn_ConvNd(nDIM)(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=3, stride=1, padding=1,
                            padding_mode='circular'),
            nn_ConvNd(nDIM)(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=3, stride=1, padding=1,
                            padding_mode='circular')
        )
        ###################################
        ### 1x1 conv  + 3x3 conv
        ###################################
        self.branch2 = nn.Sequential(
            nn_ConvNd(nDIM)(in_channels=in_fts, out_channels=out_fts[1], kernel_size=1, stride=1),
            nn_ConvNd(nDIM)(in_channels=out_fts[1], out_channels=out_fts[1], kernel_size=3, stride=1, padding=1,   padding_mode='circular'),
        )
        ###################################
        ###  3x3 MAX POOL  +  1x1 CONV
        ###################################
        self.branch3 = nn.Sequential(
            #nn_MaxPoolNd(nDIM)(kernel_size=3, stride=1, padding=1),
            #nn_ConvNd(nDIM)(in_channels=in_fts, out_channels=out_fts[2], kernel_size=1, stride=1)
            nn_CircularPadNd(nDIM)(1),                                                                                             # RY add the 'circlar' pad, April 9, 2024
            nn_MaxPoolNd(nDIM)(kernel_size=3, stride=1, padding=0),                                                          # RY add the 'circlar' pad, April 9, 2024
            nn_ConvNd(nDIM)(in_channels=in_fts, out_channels=out_fts[2], kernel_size=1, stride=1, padding_mode='circular')   # RY add the 'circlar' pad, April 9, 2024
        )
        ###################################
        ###  1x1 CONV
        ###################################
        self.branch4 = nn.Sequential(
            nn_ConvNd(nDIM)(in_channels=in_fts, out_channels=out_fts[3], kernel_size=1, stride=1, padding_mode='circular')  #  RX correction of 'circlar' pad, April 9, 2024
        )

    def forward(self, input):
        o1 = self.branch1(input)
        o2 = self.branch2(input)
        o3 = self.branch3(input)
        o4 = self.branch4(input)
        x = torch.cat([o1, o2, o3, o4], dim=-1 - self.nDIM)
        return x


# ---------------------------
class ResidualBlockNd(nn.Module):
    def __init__(self, nDIM, numRepeat, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 padding_mode='circular', bias=True, bRelu=True, bNorm=0 ):
        super(ResidualBlockNd, self).__init__()
        self.nDIM = nDIM
        self.numRepeat = numRepeat

        self.bRelu = bRelu
        self.bNorm = bNorm

        layers = []
        layers.append(
            nn_ConvNd(nDIM)(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode,
                            bias=bias))

        if bNorm == True:     layers.append( my_NormNd(nDIM,out_channels,bNorm) )

        if bRelu == True:     layers.append(nn.ReLU())

        self.cnn1 = nn.Sequential(*layers)

        # self.cnn1 =nn.Sequential(
        #    nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding, padding_mode=padding_mode, bias=bias),
        #    nn.ReLU(),
        #    nn.BatchNorm1d(out_channels),
        # )
        # self.cnn1.apply(init_weights)

        layers = []
        layers.append(  nn_ConvNd(nDIM)(out_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode,   bias=bias))
        if bNorm:     layers.append( my_NormNd(nDIM,out_channels,bNorm)  )

        self.cnn2 = nn.Sequential(*layers)
        # self.cnn2 = nn.Sequential(
        #    nn.Conv1d(out_channels,out_channels,kernel_size,     1,padding,padding_mode=padding_mode, bias=bias),
        #    nn.BatchNorm1d(out_channels)
        # )

        # if stride != 1 or in_channels != out_channels:
        #    self.shortcut = nn.Sequential(
        #        nn.Conv1d(in_channels,out_channels,kernel_size=1,stride=stride,bias=bias),
        #        #nn.BatchNorm1d(out_channels)
        #    )

        if stride != 1 or in_channels != out_channels:

            layers = []
            layers.append(nn_ConvNd(nDIM)(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                          padding_mode=padding_mode, bias=bias))
            if bNorm:     layers.append(  my_NormNd(nDIM,out_channels,bNorm)  )

            self.shortcut = nn.Sequential(*layers)

            # self.shortcut = nn.Sequential(
            #        nn.Conv1d(in_channels,out_channels,kernel_size=3,padding=1, padding_mode=padding_mode, stride=stride,bias=bias),
            #        nn.BatchNorm1d(out_channels)
            #      )

        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):

        for dummy in range(self.numRepeat):
            residual = x
            x = self.cnn1(x)
            x = self.cnn2(x)
            x += self.shortcut(residual)
            if self.bRelu:
                x = nn.ReLU()(x)

        return x

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

#----------------------------------------

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight) #, gain=nn.init.calculate_gain('relu'))
        #m.weight.data.fill_(1.0)
        #print('init weights:', m)
    elif type(m) == nn.Conv1d:
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        #print('init weights:', m)
