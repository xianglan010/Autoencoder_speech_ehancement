import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.utils as nnu
import torch.nn as nn
import random
import numpy as np
from module import *

class AE_SE(nn.Module):
    """ auto encoder based speech enhancement"""
    def __init__(self, inputs, fmaps, kwidth, poolings, dec_fmaps, dec_kwidth, dec_poolings,bias,norm_type=None,name='AE_SE'):

        super(AE_SE, self).__init__()

        # encoder
        self.enc_blocks = nn.ModuleList()
        in_layer = inputs

        # 1,64,4,31
        for i, (fmap,pool,kw) in enumerate(zip(fmaps, poolings, kwidth),start=1):
            enc_block= GConv1DBlock(in_layer,fmap,kw,stride=pool,bias=bias,norm_type=norm_type)
            self.enc_blocks.append(enc_block)
            in_layer = fmap

        # decoder 
        self.dec_blocks = nn.ModuleList()
        for i, (fmap,pool,kw) in enumerate(zip(dec_fmaps,dec_poolings,dec_kwidth),start=1):
            if i >= len(dec_fmaps):
                act = 'Tanh'
            else:
                act = None
            dec_block = GDeconv1DBlock(in_layer,fmap,kw,stride=pool,norm_type=norm_type,bias=bias,act=act)
            self.dec_blocks.append(dec_block)
            in_layer = fmap

    def forward(self, x):

        hi = x

        for l_i, enc_layer in enumerate(self.enc_blocks):
            hi, linear_hi = enc_layer(hi, True)

        enc_layer_idx = len(self.enc_blocks) - 1

        for l_i, dec_layer in enumerate(self.dec_blocks):
            hi = dec_layer(hi)
            enc_layer_idx -= 1
        return hi


