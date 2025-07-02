"""
Fixed SimVP modules with GroupNorm to handle temporal flattening
Replaces BatchNorm2d to avoid channel mismatch when T*C != C
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_num_groups(channels, base_groups=8):
    """Calculate appropriate number of groups for GroupNorm"""
    if channels <= 0:
        return 1
    
    # Find the largest divisor of channels that's <= base_groups
    for groups in range(min(base_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1  # Fallback


class BasicConv2d_GN(nn.Module):
    """Basic conv block with GroupNorm instead of BatchNorm2d"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, upsampling=False, act_norm=False, act_inplace=True):
        super(BasicConv2d_GN, self).__init__()
        self.act_norm = act_norm
        
        if upsampling:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                         stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, dilation=dilation)

        if self.act_norm:
            num_groups = get_num_groups(out_channels)
            self.norm = nn.GroupNorm(num_groups, out_channels)
            self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC_GN(nn.Module):
    """ConvSC block with GroupNorm to handle temporal flattening"""
    
    def __init__(self, C_in, C_out, stride, transposed=False, act_norm=True):
        super(ConvSC_GN, self).__init__()
        
        if transposed:
            self.conv = nn.ConvTranspose2d(C_in, C_out, kernel_size=stride, 
                                         stride=stride, padding=0)
        else:
            self.conv = nn.Conv2d(C_in, C_out, kernel_size=stride, 
                                stride=stride, padding=0)
        
        if act_norm:
            num_groups = get_num_groups(C_out)
            self.norm = nn.GroupNorm(num_groups, C_out)
            self.act = nn.SiLU()
        else:
            self.norm = None
            self.act = None

    def forward(self, x):
        y = self.conv(x)
        if self.norm is not None:
            y = self.act(self.norm(y))
        return y


class GroupConv2d_GN(nn.Module):
    """Grouped convolution with GroupNorm"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, groups=1, act_norm=False, act_inplace=True):
        super(GroupConv2d_GN, self).__init__()
        self.act_norm = act_norm
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, groups=groups)
        
        if act_norm:
            num_groups = get_num_groups(out_channels)
            self.norm = nn.GroupNorm(num_groups, out_channels)
            self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class gInception_ST_GN(nn.Module):
    """Spatial-Temporal gated Inception block with GroupNorm"""
    
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7,11], groups=8):        
        super(gInception_ST_GN, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d_GN(C_hid, C_out, kernel_size=ker, stride=1,
                                       padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y


class Mid_Xnet_GN(nn.Module):
    """Middle network with GroupNorm - properly handles temporal flattening"""
    
    def __init__(self, channel_in, channel_hid, N_T, incep_ker=[3,5,7,11], groups=8):
        super(Mid_Xnet_GN, self).__init__()

        assert N_T >= 2
        self.N_T = N_T

        # Encoder layers
        enc_layers = [gInception_ST_GN(channel_in, channel_hid//2, channel_hid, 
                                      incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(gInception_ST_GN(channel_hid, channel_hid//2, channel_hid, 
                                             incep_ker=incep_ker, groups=groups))
        enc_layers.append(gInception_ST_GN(channel_hid, channel_hid//2, channel_hid, 
                                         incep_ker=incep_ker, groups=groups))

        # Decoder layers
        dec_layers = [gInception_ST_GN(channel_hid, channel_hid//2, channel_hid, 
                                      incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(gInception_ST_GN(2*channel_hid, channel_hid//2, channel_hid, 
                                             incep_ker=incep_ker, groups=groups))
        dec_layers.append(gInception_ST_GN(2*channel_hid, channel_hid//2, channel_in, 
                                         incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # Temporal flattening: (B, T, C, H, W) -> (B, T*C, H, W)
        # This is where BatchNorm2d fails because it expects C channels but gets T*C
        x = x.reshape(B, T*C, H, W)
        
        skips = []
        z = x
        for i, layer in enumerate(self.enc):
            z = layer(z)
            if i < self.N_T - 1:
                skips.append(z)

        z = self.dec[0](z)
        for i in range(1, self.N_T):
            if i <= len(skips):
                z = self.dec[i](torch.cat([z, skips[-i]], dim=1))
            else:
                z = self.dec[i](z)

        # Reshape back: (B, T*C, H, W) -> (B, T, C, H, W)
        y = z.reshape(B, T, C, H, W)
        return y


class Encoder_GN(nn.Module):
    """Encoder with GroupNorm"""
    
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder_GN, self).__init__()
        strides = [1, 2, 2, 2]
        
        self.enc = nn.Sequential(
            ConvSC_GN(C_in, C_hid, stride=strides[0]),
            *[ConvSC_GN(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder_GN(nn.Module):
    """Decoder with GroupNorm"""
    
    def __init__(self, C_hid, C_out, N_S):
        super(Decoder_GN, self).__init__()
        strides = [2, 2, 2, 1]

        self.dec = nn.Sequential(
            *[ConvSC_GN(C_hid, C_hid, stride=s, transposed=True) for s in strides[:-1]],
            ConvSC_GN(C_hid, C_out, stride=strides[-1])
        )

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid)
        return Y


class SimVP_Model_GN(nn.Module):
    """
    SimVP Model with GroupNorm - fixes temporal flattening issues
    
    This version replaces all BatchNorm2d with GroupNorm to handle the case where
    temporal dimensions are flattened from (B, T, C, H, W) to (B, T*C, H, W).
    BatchNorm2d expects C channels but gets T*C channels, causing shape mismatches.
    """
    
    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=8, 
                 incep_ker=[3,5,7,11], groups=8, **kwargs):
        super(SimVP_Model_GN, self).__init__()
        T, C, H, W = in_shape
        
        self.enc = Encoder_GN(C, hid_S, N_S)
        self.hid = Mid_Xnet_GN(hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder_GN(hid_S, C, N_S)

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)  # This is where temporal flattening happens
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid)
        Y = Y.reshape(B, T, C, H, W)
        return Y


# For easy import replacement
def create_simvp_groupnorm_model(in_shape, **kwargs):
    """Factory function to create GroupNorm-based SimVP model"""
    return SimVP_Model_GN(in_shape=in_shape, **kwargs)