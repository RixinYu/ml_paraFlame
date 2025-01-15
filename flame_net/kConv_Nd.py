import torch
import torch.nn as nn
#import numpy as np

from flame_net.MyConvNd import MyConvNd, nn_MaxPoolNd

#
#  (N-dimentional) Koopman theory inspired Convolutional Neural Network (kCNN)
#  [Yu, R., Herbert, M., Klein, M. and Hodzic, E., 2024. Koopman Theory-Inspired Method for Learning Time Advancement Operators in Unstable Flame Front Evolution. arXiv preprint arXiv:2412.08426.]
# 
class kConv_Nd(nn.Module):
    def __init__(self, nDIM, N, in_channel=1, kTimeStepping=20,
                 en1_channels =[ 16,[32,32],[64,64],[64],[64] ],
                 de1_channels = None,
                 method_types_conv ='inception_less',
                 method_BatchNorm=0,
                 method_outputTanh=None,
                 out_channel=1  ):
        super(kConv_Nd, self).__init__()

        self.nDIM = nDIM
        self.kTimeStepping = kTimeStepping
        self.N = N
        self.in_channel = in_channel
        if out_channel is None:         out_channel = in_channel
        self.out_channel = out_channel
        if method_outputTanh is not None:        self.method_outputTanh = method_outputTanh


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
            first_channel = in_channel  if l==0 else en_channels[l-1][-1]
            en_channels[l] = [first_channel] + en_channels[l]
        self.en_channels  = en_channels
        #---------------

        de_channels = de1_channels[:] # make a copy
        for l, _ in enumerate(de_channels):
            first_channel = en_channels[-1][-1] if l==0 else de_channels[l-1][-1]
            if l>= 1:   first_channel = first_channel*2   # '*2' is for the skip connection
            de_channels[l] = [first_channel] + de_channels[l]
        self.de_channels  = de_channels

        #--------------
        self.tAdv_channels  = []
        for  ly in self.en_channels:  self.tAdv_channels.append( [ly[-1],ly[-1],ly[-1] ] )
        self.tAdv_channels.append( [ly[-1],ly[-1],ly[-1] ] )
        #-------------

        print( 'kConv_Nd: en_channels = ',   self.en_channels )
        print( 'kConv_Nd: tAdv_channels = ', self.tAdv_channels )
        print( 'kConv_Nd: de_channels = ',   self.de_channels )


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

        self.en_types =en_types
        self.de_types =de_types
        print('kConv_Nd: en_types = ', en_types)
        print('kConv_Nd: de_types = ', de_types)

        #--------------------------------
        PoolNd = nn_MaxPoolNd(nDIM)(kernel_size=2, stride=2)
        #---------------
        self.en_conv= nn.ModuleList()
        for l , channels_list in enumerate(self.en_channels) :
            layers =[]
            if l >0:    layers.append( PoolNd )
            my_bNorm = method_BatchNorm if method_BatchNorm<=0 else method_BatchNorm//2**l

            for idx in range( len(channels_list)-1):
                _my_bNorm_ = my_bNorm
                layers.append( MyConvNd(nDIM, channels_list[idx],channels_list[idx+1],kernel_size=3, type=self.en_types[l][idx], bRelu= True , bNorm=_my_bNorm_ )  )
            self.en_conv.append( nn.Sequential( *layers ) )
        self.en_conv.append( PoolNd )
        #----------------
        self.de_conv = nn.ModuleList()
        for l, channels_list in enumerate( self.de_channels):
            layers = []
            # if l ==0: layers.append( PoolNd )
            for idx in range( len(channels_list)-1 ) :
                bRelu_set0            = False if l==len(self.de_channels)-1 and idx==len(channels_list)-2 else True
                method_BatchNorm_set0 = 0     if l==len(self.de_channels)-1 and idx==len(channels_list)-2 else method_BatchNorm
                my_bNorm= method_BatchNorm_set0 if method_BatchNorm_set0<=0 else method_BatchNorm_set0//2**(len(self.en_channels)-l)
                layers.append( MyConvNd(nDIM, channels_list[idx] , channels_list[idx+1], kernel_size=3,  type=self.de_types[l][idx], bRelu=bRelu_set0, bNorm=my_bNorm)  )

            if l< len(self.de_channels)-1:      layers.append(nn.Upsample(scale_factor=2))
            #-----------
            self.de_conv.append( nn.Sequential( *layers ) )




        self.TimeAdv_conv = nn.ModuleList()
        for l, channels_list in enumerate( self.tAdv_channels):
            layers = []
            for idx in range( len(channels_list)-1 ) :
                bRelu_set0  = True if idx==0 else False
                layers.append( MyConvNd(nDIM, channels_list[idx] , channels_list[idx+1], kernel_size=3,  type='c', bRelu=bRelu_set0, bNorm=0 )  )
            self.TimeAdv_conv.append( nn.Sequential( *layers ) )

    def encoder(self,x):
        x_all = []
        for l, conv in enumerate( self.en_conv ):
            x_en = conv(x)
            x_all.append(x_en)
            x = x_en
        return x_all

    def decoder(self, x_all ):
        x = x_all[-1]
        for l, conv in enumerate(self.de_conv):
            if l==0:
                x = conv( x )
            else:
                x = torch.cat( (x, x_all[-1-l]), dim=-1-self.nDIM )
                x = conv( x )
        return x

    def TimeAdv(self, x_all ):
        for j, (x, adv_conv) in enumerate( zip(x_all, self.TimeAdv_conv) ):
            x_all[j] = x + adv_conv(x)
        return x_all


    def forward(self, x , p=None):

        batchsize= x.shape[0]

        x_o = torch.zeros( *x.shape[:-1],self.out_channel, self.kTimeStepping, device=x.device ) # b,(Nx,Ny),out,t

        if self.nDIM==1:    x = x.permute(0, 2, 1)
        elif self.nDIM==2:  x = x.permute(0, 3, 1, 2)

        x_all = self.encoder( x )

        for i in range(self.kTimeStepping):
            if i>0:
                x_all = self.TimeAdv(x_all)

            x  = self.decoder(x_all)

            if self.nDIM==1:    x = x.permute(0, 2, 1)
            elif self.nDIM==2:  x = x.permute(0, 2, 3, 1)

            if self.kTimeStepping==1:
                return x

            x_o[...,i] = x

        if self.kTimeStepping == 1:             x_o = x_o.view( *x_o.shape[:-1] )
        if hasattr(self,'method_outputTanh'):   x_o = torch.tanh( x_o )

        return x_o







#########################################################################
#
#
#
#
#
#   The following may be deleted
#
#
#
#
#########################################################################

def mm_ConvNd(nDIM,i,o,nFilter=3,padding=1):
    if nDIM==1:          return nn.Conv1d(i,o,nFilter,padding_mode='circular',padding=padding )
    elif nDIM==2:        return nn.Conv2d(i,o,nFilter,padding_mode='circular',padding=padding)
    else:                raise ValueError('mm_ConvNd: nDIM='+str(nDIM) )

def mm_MaxPoolNd(nDIM):
    if nDIM==1:        return nn.MaxPool1d(kernel_size=2, stride=2)
    elif nDIM==2:      return nn.MaxPool2d(kernel_size=2, stride=2)
    else:              raise ValueError('mm_MaxPoolNd: nDIM='+str(nDIM) )

class ResN(nn.Module):
    def __init__(self,nDIM,i, o ,bRelu):
        super(ResN, self).__init__()
        self.net = mm_ConvNd(nDIM,i,o,3)
        self.bRelu = bRelu
    def forward(self,x):
        if self.bRelu: return x + nn.GELU()(self.net(x))
        else:          return x +           self.net(x)

class CNull(nn.Module):
    def __init__(self):     super(CNull, self).__init__()
    def forward(self,x):    return x

class kCNN_Nd(nn.Module):
    def __init__(self, nDIM=1,in_channel=1, out_channel=1, kTimeStepping =20 ):
        super(kCNN_Nd, self).__init__()
        self.nDIM = nDIM
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kTimeStepping = kTimeStepping
        #----------
        PoolNd = mm_MaxPoolNd(nDIM)
        ConvNd = lambda i,o : mm_ConvNd(nDIM,i,o,3)
        RConvNd = lambda i,o,T_F : ResN(nDIM,i,o,T_F)
        Ln      = lambda c, w: nn.LayerNorm( [c,w ] )
        Ln0      = lambda c, w: CNull()
        #Ln      = lambda c,w: torch.nn.BatchNorm1d(c)

        #----------
        en_l = [ [       ConvNd(in_channel,32), nn.ReLU(), RConvNd(32,32,True), Ln(32,64)], #64
                 [PoolNd,ConvNd(32,64), nn.ReLU(), RConvNd(64,64,True),  Ln(64,32)],         #32
                 [PoolNd,ConvNd(64,128), nn.ReLU(), RConvNd(128,128,True), Ln(128,16)],         #16
                 [PoolNd,RConvNd(128,128,True),     RConvNd(128,128,True), Ln(128,8)],              #8
                 [PoolNd,RConvNd(128,128,True)  ] ]             #4

        de_l = [ [RConvNd(128,128,True), Ln(128,4), nn.Upsample(scale_factor=2)] ,
                 [ConvNd(2*128,128), nn.ReLU(),RConvNd(128,128,True), Ln(128,8), nn.Upsample(scale_factor=2)],
                 [ConvNd(2*128,128),nn.ReLU(),ConvNd(128,64),nn.ReLU(), Ln(64,16), nn.Upsample(scale_factor=2)], #8
                 [ConvNd(2*64,64), nn.ReLU(),ConvNd(64,32),nn.ReLU(), Ln(32,32), nn.Upsample(scale_factor=2)],
                 [ConvNd(2*32,32),  nn.ReLU(), Ln(32,64), ConvNd(32,out_channel)]   ] #2

        tAdv_l = [ [RConvNd(32,32,True), RConvNd(32,32,False) , Ln0(32,64)] , #256
                 [RConvNd(64,64,True) , RConvNd(64,64,False), Ln0(64,32) ], #128
                 [RConvNd(128,128,True) ,RConvNd(128,128,False), Ln0(128,16)],  #64
                 [RConvNd(128,128,True) ,RConvNd(128,128,False), Ln0(128,8)],    #4
                 [RConvNd(128,128,True) ,RConvNd(128,128,False), Ln0(128,4)] ] #2
        # #----------
        # en_l = [ [       ResNet(nDIM,[in_channel,16,16]) ], #256
        #          [PoolNd,ResNet(nDIM,[16,32,32]) ],         #128
        #          [PoolNd,ResNet(nDIM,[32,64,64]) ],         #64
        #          [PoolNd,ResNet(nDIM,[64,64,64]) ],        #8
        #          [PoolNd,ResNet(nDIM,[64,64,64]) ],         #4
        #          [PoolNd,ResNet(nDIM,[64,64]   )] ]                                #2
        # de_l = [ [ResNet(nDIM,[64,64]), nn.Upsample(scale_factor=2)], #2
        #          [ResNet(nDIM,[2*64,64,64]),nn.Upsample(scale_factor=2)], #4
        #          [ResNet(nDIM,[2*64,64,64]),nn.Upsample(scale_factor=2)], #8
        #          [ResNet(nDIM,[2*64,32,32]),nn.Upsample(scale_factor=2)],
        #          [ResNet(nDIM,[2*32,16,16]),nn.Upsample(scale_factor=2)],
        #          [ResNet(nDIM,[2*16,32,out_channel],False) ]  ]

        # tAdv_l =[[ResNet(nDIM,[16,16,16]) ] , #256
        #          [ResNet(nDIM,[32,32,32]) ], #128
        #          [ResNet(nDIM,[64,64,64]) ],  #64
        #          [ResNet(nDIM,[64,64,64]) ],    #8
        #          [ResNet(nDIM,[64,64,64]) ],    #4
        #          [ResNet(nDIM,[64,64,64]) ] ] #2

        # en_l = [ [ConvNd(in_channel,16),nn.GELU(),ConvNd(16,16), nn.GELU()], #256
        #          [ConvNd(16,32),nn.GELU(),ConvNd(32,32), nn.GELU()],         #128
        #          [ConvNd(32,64),nn.GELU(),ConvNd(64,64), nn.GELU()],         #64
        #          [ConvNd(64,128),nn.GELU(),ConvNd(128,128),nn.GELU()],       #32
        #          [ConvNd(128,128),nn.GELU(),ConvNd(128,128),nn.GELU()],      #16
        #          [ConvNd(128,64),nn.GELU(),ConvNd(64,64), nn.GELU()],        #8
        #          [ConvNd(64,64),nn.GELU(),ConvNd(64,64), nn.GELU()],         #4
        #          [ConvNd(64,64),nn.GELU() ] ]                                #2

        # de_l = [ [ConvNd(64,64),nn.GELU(),nn.Upsample(scale_factor=2)], #2
        #          [ConvNd(2*64,64),nn.GELU(),ConvNd(64,64), nn.GELU(),nn.Upsample(scale_factor=2)],    #4
        #          [ConvNd(2*64,128),nn.GELU(),ConvNd(128,128), nn.GELU(),nn.Upsample(scale_factor=2)], #8
        #          [ConvNd(2*128,128),nn.GELU(),ConvNd(128,128),nn.GELU(),nn.Upsample(scale_factor=2)], #16
        #          [ConvNd(2*128,64),nn.GELU(),ConvNd(64,64),nn.GELU(),nn.Upsample(scale_factor=2)],    #32
        #          [ConvNd(2*64,32),nn.GELU(),ConvNd(32,32),nn.GELU(),nn.Upsample(scale_factor=2)],     #64
        #          [ConvNd(2*32,16),nn.GELU(),ConvNd(16,16), nn.GELU(),nn.Upsample(scale_factor=2)],    #128
        #          [ConvNd(2*16,16),nn.GELU(),ConvNd(16,out_channel) ] ]                                #256

        # tAdv_l = [ [ConvNd(16,16),nn.GELU(),ConvNd(16,16)], #256
        #          [ConvNd(32,32),nn.GELU(),ConvNd(32,32)], #128
        #          [ConvNd(64,64),nn.GELU(),ConvNd(64,64)],  #64
        #          [ConvNd(128,128),nn.GELU(),ConvNd(128,128)],#32
        #          [ConvNd(128,128),nn.GELU(),ConvNd(128,128)],#16
        #          [ConvNd(64,64),nn.GELU(),ConvNd(64,64)],    #8
        #          [ConvNd(64,64),nn.GELU(),ConvNd(64,64)],    #4
        #          [ConvNd(64,64),nn.GELU(),ConvNd(64,64)  ] ] #2

        self.en_conv= nn.ModuleList()
        self.de_conv= nn.ModuleList()
        self.TimeAdv_conv= nn.ModuleList()
        for l in en_l:            self.en_conv.append( nn.Sequential( *l ) )
        for l in de_l:            self.de_conv.append( nn.Sequential( *l ) )
        for l in tAdv_l:          self.TimeAdv_conv.append( nn.Sequential( *l ) )
        return

    def encoder(self,x):
        x_all = []
        for l, conv in enumerate( self.en_conv ):
            x_en = conv(x)
            x_all.append(x_en)
            x = x_en
        return x_all

    def decoder(self, x_all ):
        #x_all = x_all[::-1]  #reverse
        x = x_all[-1]
        for l, conv in enumerate(self.de_conv):
            if l==0:
                x = conv( x )
            else:  #  and  l-1<len(x_all) :

                #print('l',l, 'x.shape', x.shape,'x_all[-1].shape', x_all[-1].shape, 'x_all[-1-l].shape', x_all[-1-l].shape, 'conv', conv)

                x = torch.cat( (x, x_all[-1-l]), dim=-1-self.nDIM )

                #print('l',l, 'x.shape', x.shape,'x_all[-1].shape', x_all[-1].shape, 'x_all[-1-l].shape', x_all[-1-l].shape, 'conv', conv)

                x = conv( x )
        return x

    def TimeAdv(self, x_all ):
        for j, (x, adv_conv) in enumerate( zip(x_all, self.TimeAdv_conv) ):
            if j>= -99: #  len(x_all)-2: #-99
                x_all[j] = adv_conv(x)
        return x_all

    def forward(self, x , p=None):
        '''
            x.shape = b, c, Nx, Ny
            p.shape = b, num_PDEParameters (if not None)
        '''
        batchsize= x.shape[0]

        x_o = torch.zeros( *x.shape[:-1], self.out_channel, self.kTimeStepping, device=x.device ) # b,(Nx,Ny),out,t

        if self.nDIM==1:    x = x.permute(0, 2, 1)
        elif self.nDIM==2:  x = x.permute(0, 3, 1, 2)

        x_all = self.encoder( x )

        for i in range(self.kTimeStepping):

            if i>0:
                x_all = self.TimeAdv(x_all)

            x = self.decoder(x_all)

            if self.nDIM==1:    x = x.permute(0, 2, 1)
            elif self.nDIM==2:  x = x.permute(0, 2, 3, 1)

            if self.kTimeStepping == 1:
                return x

            x_o[...,i] = x

        return x_o
