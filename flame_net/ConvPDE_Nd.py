


#-----------------------------------------------------
#import torch.nn.functional as F
#from  torch.autograd.functional import vjp
#import operator
#from functools import reduce
#from functools import partial
#from utilities3 import *
#-----------------------------------------------------

import torch
import torch.nn as nn
import numpy as np

from flame_net.MyConvNd import MyConvNd, nn_MaxPoolNd, nn_AvgPoolNd , Fc_PDEpara  # ,nn_BatchNormNd



class ConvPDE_Nd(nn.Module):
    def __init__(self, nDIM, N,
                 in_channel=1, out_channel=1,
                 en1_channels =[ 8,16,32,64,[64] ], #  None, [,,,,]
                 de1_channels = None,
                 method_nonlinear='all', # 'all', 'none', 'de'(i.e. decoder)
                 method_types_conv ='conv_all',
                 #method_OP='', #this key is removed: ( #'none', 'OP')
                 method_skip='full', # 'off', #   (decaperated:  'width2', 'width4' )
                 bUpSampleOrConvTranspose='upsample',
                 method_pool='max', # 'ave'
                 #method_conv='',       #this key is removed!!!! # 'more' # no allowed( 'share')
                 num_PDEParameters=0 ,
                 method_BatchNorm=0,
                 PDEPara_depth =  None , # or None
                 method_ParaEmbedding = False
                 #bExternalConstraint = False
                 #,yB_1DNormalization =None
                 ):
        super(ConvPDE_Nd, self).__init__()

        #self.bExternalConstraint = bExternalConstraint


        self.nDIM = nDIM
        self.N = N
        self.in_channel = in_channel
        if out_channel is None:
            out_channel = in_channel

        self.out_channel = out_channel
        self.num_PDEParameters=num_PDEParameters

        #----------
        if PDEPara_depth is None and method_ParaEmbedding==False and num_PDEParameters >0:
            assert False, "wrong setup calling pCNN"
        self.method_ParaEmbedding = method_ParaEmbedding
        #----------


        if 'off' in method_skip.casefold(): #
            self.bSkipConnect = False
        else:
            self.bSkipConnect = True

        # The following convert [3,[5,8],7] to [[3],[5,8],[7]]
        for idx, c in enumerate( en1_channels):
            try: # check if it is a list
                c[0]
            except:
                en1_channels[idx] = [c]

        if de1_channels is None:
            de1_channels =  en1_channels[::-1]  + [[out_channel]]  # (i) skip the last then reverse, later get the last element
            for idx, c_l in enumerate(de1_channels):
                de1_channels[idx] = [ c_l[-1] ]

        #---------------
        en_channels   =  en1_channels[:] #make a copy
        for l, _ in enumerate(en_channels ):
            if self.method_ParaEmbedding == True :
                first_channel = in_channel + self.in_channel if l==0 else en_channels[l-1][-1]
            else:
                first_channel = in_channel                   if l==0 else en_channels[l-1][-1]

            en_channels[l] = [first_channel] + en_channels[l]
        self.en_channels  = en_channels
        #---------------

        de_channels = de1_channels[:] #make a copy
        for l, _ in enumerate(de_channels):
            first_channel = en_channels[-1][-1] if l==0 else de_channels[l-1][-1]
            if l>= 1:
                first_channel = first_channel*2   # '*2' is for the skip connection
            de_channels[l] = [first_channel] + de_channels[l]
        self.de_channels  = de_channels

        print( 'ConvPDE_Nd: en_channels = ', self.en_channels )
        print( 'ConvPDE_Nd: de_channels = ', self.de_channels )

        # --------------------
        #self.yB_1DNormalization = None
        #if yB_1DNormalization is not None:
        #    self.yB_1DNormalization = torch.tensor( yB_1DNormalization, dtype=torch.float )

        #------------------------
        self.method_types_conv = method_types_conv

        type_char =  method_types_conv[0]
        en_types = [ [type_char for _ in range(len(c_l)) ] for c_l in en1_channels]
        de_types = [ [type_char for _ in range(len(c_l)) ] for c_l in de1_channels]

        if 'inception_most' in method_types_conv.casefold():
            de_types[-1][-1] = 'c'
        elif 'inception_less' in method_types_conv.casefold():
            en_types = [['c' for _ in range(len(c_l))] for c_l in en1_channels]
            de_types = [['c' for _ in range(len(c_l))] for c_l in de1_channels]
            if len(en_types[0])>1:
                en_types[0][-1] = 'i'
            en_types[ 1][-1] = 'i'
            en_types[ 2][-1] = 'i'
            en_types[ 3][-1] = 'i'

            de_types[-2][-1] = 'i'
        #else:
        #    raise ValueError( method_types_conv + ', wrong method_types_conv')

        self.en_types =en_types
        self.de_types =de_types
        print('ConvPDE_Nd: en_types = ', en_types)
        print('ConvPDE_Nd: de_types = ', de_types)

        #--------------------------------
        #self.method_conv = method_conv
        self.method_pool = method_pool
        if 'max' in method_pool.casefold():
            PoolNd = nn_MaxPoolNd(nDIM)(kernel_size=2, stride=2)
        else: #'ave'
            PoolNd = nn_AvgPoolNd(nDIM)(kernel_size=2, stride=2)


        bRelu_skip = False

        #self.method_skip = method_skip

        self.bRelu_Conv = True

        self.bUpSampleOrConvTranspose = bUpSampleOrConvTranspose


        self.bRelu_Conv0__en = True if 'all' in method_nonlinear.casefold() else False
        self.bRelu_Conv0__de = True if ('all' in method_nonlinear.casefold() or 'de' in method_nonlinear.casefold() ) else False

        #---------------
        self.en_conv0= nn.ModuleList()

        for l , channels_list in enumerate(self.en_channels) :
            layers =[]
            if l >0:
                layers.append( PoolNd )

            my_bNorm= method_BatchNorm if method_BatchNorm<=0 else method_BatchNorm//2**l

            for idx in range( len(channels_list)-1):
                layers.append( MyConvNd(nDIM, channels_list[idx],channels_list[idx+1],kernel_size=3, type=self.en_types[l][idx], bRelu=self.bRelu_Conv0__en, bNorm=my_bNorm )  )

            self.en_conv0.append( nn.Sequential( *layers ) )
        #----------------

        if self.num_PDEParameters>=1: # learning  PDEs with multi parameters
            self.PDEPara_depth = PDEPara_depth
            # if method_BatchNorm:
            #     self.fc_PDEPara = nn.Sequential(
            #                         nn.Linear( self.num_PDEParameters, 15,bias=False), nn.LayerNorm(15),  nn.ReLU(),
            #                         nn.Linear( 15, 15,bias=False),                     nn.LayerNorm(15),  nn.ReLU(),
            #                         nn.Linear( 15, self.PDEPara_depth,bias=False ),                       nn.Tanh(),
            #                       )
            # else:
            #     self.fc_PDEPara = nn.Sequential(
            #                         nn.Linear( self.num_PDEParameters, 15),  nn.ReLU(), # nn.Sigmoid(), #   nn.ReLU(),
            #                         nn.Linear( 15, 15),                      nn.ReLU(), # nn.Sigmoid(), #  nn.ReLU(),
            #                         nn.Linear( 15, self.PDEPara_depth ),     nn.Tanh()
            #                       )
            self.fc_PDEPara = Fc_PDEpara(self.num_PDEParameters, self.PDEPara_depth ) #, method_BatchNorm  )

            self.en_conv_PDEPara = nn.ModuleList()
            for l in range( self.PDEPara_depth ):
                layers_ex = []
                if l >0:      layers_ex.append( PoolNd )

                my_bNorm= method_BatchNorm if method_BatchNorm<=0 else method_BatchNorm//2**l


                layers_ex.append( MyConvNd( nDIM, en_channels[l][0] , en_channels[l][-1], kernel_size = 3, bNorm=my_bNorm ) )
                self.en_conv_PDEPara.append(  nn.Sequential( *layers_ex )  )


        #---------------
        self.de_conv0 = nn.ModuleList()
        for l, channels_list in enumerate( self.de_channels):
            layers = []
            if l ==0:
                layers.append( PoolNd )

            if self.bUpSampleOrConvTranspose =='upsample':

                for idx in range( len(channels_list)-1 ) :

                    bRelu_set0            = False if l==len(self.de_channels)-1 and idx==len(channels_list)-2 else self.bRelu_Conv0__de
                    method_BatchNorm_set0 = 0 if l==len(self.de_channels)-1 and idx==len(channels_list)-2 else method_BatchNorm

                    my_bNorm= method_BatchNorm_set0 if method_BatchNorm_set0<=0 else method_BatchNorm_set0//2**(len(self.en_channels)-l)


                    layers.append( MyConvNd(nDIM, channels_list[idx] , channels_list[idx+1], kernel_size=3,  type=self.de_types[l][idx], bRelu=bRelu_set0, bNorm=my_bNorm)  )

                if l< len(self.de_channels)-1:
                    layers.append(nn.Upsample(scale_factor=2))

            # else: # 'ConvTranspose' instead of 'upsample'
            #     for idx in range( len(channels_list)-2 ) :
            #         bRelu_set0 = self.bRelu_Conv0__de
            #         layers.append( MyConvNd(nDIM, channels_list[idx] , channels_list[idx+1], kernel_size=3,  type=self.de_types[l][idx], bRelu=bRelu_set0, bNorm=method_BatchNorm)  )
            #
            #     if l == len(self.de_channels)-1:
            #         bRelu_set0 = False
            #         type_set = self.de_types[l][len(channels_list)-2]
            #         layers.append( MyConvNd(nDIM, channels_list[-2], channels_list[-1], kernel_size=3,  type=type_set,   bRelu=bRelu_set0, bNorm=method_BatchNorm))
            #     elif l< len(self.de_channels)-1:
            #         bRelu_set0 = self.bRelu_Conv0__de
            #         type_set = 'Transpose'
            #         layers.append(MyConvNd(nDIM, channels_list[-2], channels_list[-1], kernel_size=2, stride=2, type=type_set, bRelu=bRelu_set0, bNorm=method_BatchNorm))


            self.de_conv0.append( nn.Sequential( *layers ) )


    def encoder(self,x, weight_scaling = None ):
        x_all = []

        for l, _ in enumerate(self.en_conv0):
            x_en = self.en_conv0[l](x)

            #if weight_scaling is not None:
            if self.num_PDEParameters >=1:
                if l < len(self.en_conv_PDEPara):
                    x_en = x_en + weight_scaling[:,l].view(-1, *((self.nDIM+1)*[1]) ) * self.en_conv_PDEPara[l](x)
                    #                                 view(-1,1,1,1)

            x_all.append(x_en)

            x = x_en

        return x_all

    def decoder(self, x_all ):

        x_all = x_all[::-1]  #reverse
        x = x_all[0]


        for l, _ in enumerate(self.de_conv0):
            if self.bSkipConnect==True and l>=1 and  l-1<len(x_all) :
                #print('l', l , x.shape, x_all[l].shape )
                x = self.de_conv0[l]( torch.cat( (x, x_all[l-1]), dim=-1-self.nDIM )   )
                #x = self.share_conv_at_level(x,l,'de')
            else:
                x = self.de_conv0[l]( x )
                #x = self.share_conv_at_level(x,l,'de')
        return x


    def forward(self, x, p = None ):
        '''
            x.shape = b, c, Nx, Ny
            p.shape = b, num_PDEParameters (if not None)
        '''

        #if self.yB_1DNormalization is not None:
        #    x = 2*( x - self.yB_1DNormalization[0] )/( self.yB_1DNormalization[1] - self.yB_1DNormalization[0] ) - 1

        batchsize= x.shape[0]

        #----
        weight_scaling = None
        if p is not None:
            p = p.view(batchsize, self.num_PDEParameters)
            weight_scaling = self.fc_PDEPara( p  )

        #-------------------------
        #if method_ParaEmbedding == 1:
        if self.in_channel < self.en_conv0[0][0].net[0].in_channels:
            if   self.nDIM == 1:    x = torch.cat( ( x , p.unsqueeze(1)             .repeat(1,x.shape[1],           1) ) , dim=-1 )
            elif self.nDIM == 2:    x = torch.cat( ( x , p.unsqueeze(1).unsqueeze(1).repeat(1,x.shape[1],x.shape[2],1) ) , dim=-1 )
        #--------------------------


        x = x.permute(0, 2, 1) if self.nDIM == 1 else x.permute(0, 3, 1, 2)

        x_all = self.encoder(x, weight_scaling )
        x     = self.decoder(x_all)

        x = x.permute(0, 2, 1) if self.nDIM == 1 else x.permute(0, 2, 3, 1)


        #if self.yB_1DNormalization is not None:
        #    x = (x+1)/2* (self.yB_1DNormalization[1] - self.yB_1DNormalization[0]) +  self.yB_1DNormalization[0]

        #if self.bExternalConstraint ==True:
        #    if self.nDIM ==1:
        #        x =  x - torch.mean(x , dim=-2,keepdim=True)

        return x

#%%

