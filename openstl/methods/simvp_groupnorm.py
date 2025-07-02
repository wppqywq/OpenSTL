"""
Custom SimVP modules with GroupNorm instead of BatchNorm2d

This fixes the channel mismatch issue when temporal dimensions are flattened.
Use this instead of the original SimVP implementation for COCO-Search18.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.layers import DropPath
except ImportError:
    from timm.models.layers import DropPath
import math


def get_num_groups(channels, base_groups=8):
    """Calculate appropriate number of groups for GroupNorm"""
    if channels % base_groups == 0:
        return base_groups
    # Find largest divisor of channels that's <= base_groups
    for groups in range(base_groups, 0, -1):
        if channels % groups == 0:
            return groups
    return 1  # Fallback to 1 group (equivalent to LayerNorm)


class BasicConv2d(nn.Module):
    """Basic conv block with GroupNorm instead of BatchNorm"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, upsampling=False, act_norm=False, act_inplace=True):
        super(BasicConv2d, self).__init__()
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

        # Use GroupNorm instead of BatchNorm2d
        num_groups = get_num_groups(out_channels)
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    """ConvSC block with GroupNorm"""
    
    def __init__(self, C_in, C_out, stride, transposed=False, act_norm=True, 
                 gated=False, separable=False):
        super(ConvSC, self).__init__()
        self.gated = gated
        self.separable = separable
        
        if separable:
            # Depthwise separable convolution
            if transposed:
                self.conv = nn.Sequential(
                    nn.ConvTranspose2d(C_in, C_in, kernel_size=stride, stride=stride, 
                                     padding=0, groups=C_in),
                    nn.Conv2d(C_in, C_out, kernel_size=1)
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(C_in, C_in, kernel_size=stride, stride=stride, 
                             padding=0, groups=C_in),
                    nn.Conv2d(C_in, C_out, kernel_size=1)
                )
        else:
            if transposed:
                self.conv = nn.ConvTranspose2d(C_in, C_out, kernel_size=stride, 
                                             stride=stride, padding=0)
            else:
                self.conv = nn.Conv2d(C_in, C_out, kernel_size=stride, 
                                    stride=stride, padding=0)
        
        if act_norm:
            if gated:
                # For gated convolution, double the output channels
                gate_channels = C_out * 2 if not separable else C_out * 2
                num_groups = get_num_groups(gate_channels)
                self.norm = nn.GroupNorm(num_groups, gate_channels)
            else:
                num_groups = get_num_groups(C_out)
                self.norm = nn.GroupNorm(num_groups, C_out)
            self.act = nn.SiLU()
        else:
            self.norm = None
            self.act = None

    def forward(self, x):
        y = self.conv(x)
        if self.norm is not None:
            y = self.norm(y)
            if self.gated:
                # Split channels for gating
                y, gate = y.chunk(2, dim=1)
                y = y * torch.sigmoid(gate)
            y = self.act(y)
        return y


class GroupConv2d(nn.Module):
    """Grouped convolution with GroupNorm"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, groups=1, act_norm=False, act_inplace=True):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, groups=groups)
        
        if act_norm:
            num_groups = get_num_groups(out_channels)
            self.norm = nn.GroupNorm(num_groups, out_channels)
            self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class gInception_ST(nn.Module):
    """Spatial-Temporal gated Inception block with GroupNorm"""
    
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7,11], groups=8):        
        super(gInception_ST, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1,
                                    padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y


class AttentionModule(nn.Module):
    """Attention mechanism with GroupNorm"""
    
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3))
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0))

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5))
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0))

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10))
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0))
        
        # Use GroupNorm instead of BatchNorm
        num_groups = get_num_groups(dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)
        self.norm = nn.GroupNorm(num_groups, dim)

    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)
        attn = self.norm(attn)
        
        return attn * u


class SpatialAttention(nn.Module):
    """Spatial attention with GroupNorm"""
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class GASubBlock(nn.Module):
    """Gated Attention SubBlock with GroupNorm"""
    
    def __init__(self, dim, kernel_size=21, mlp_ratio=4., drop_path=0., 
                 act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.GroupNorm(get_num_groups(dim), dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.GroupNorm(get_num_groups(dim), dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1)
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TemporalAttention(nn.Module):
    """Temporal attention mechanism with GroupNorm"""
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, 
                                   stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)
        return out


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization"""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class Encoder(nn.Module):
    """Encoder with GroupNorm"""
    
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder, self).__init__()
        strides = [1, 2, 2, 2]
        
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)

        return latent, enc1


class Decoder(nn.Module):
    """Decoder with GroupNorm"""
    
    def __init__(self, C_hid, C_out, N_S):
        super(Decoder, self).__init__()
        strides = [2, 2, 2, 1]

        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transposed=True) for s in strides[:-1]],
            ConvSC(C_hid, C_out, stride=strides[-1])
        )

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid)
        return Y


class Mid_Xnet(nn.Module):
    """Middle network with GroupNorm - handles temporal flattening properly"""
    
    def __init__(self, channel_in, channel_hid, N_T, incep_ker=[3,5,7,11], groups=8):
        super(Mid_Xnet, self).__init__()

        assert N_T >= 2

        self.N_T = N_T
        enc_layers = [gInception_ST(channel_in, channel_hid//2, channel_hid, 
                                   incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(gInception_ST(channel_hid, channel_hid//2, channel_hid, 
                                          incep_ker=incep_ker, groups=groups))

        enc_layers.append(gInception_ST(channel_hid, channel_hid//2, channel_hid, 
                                       incep_ker=incep_ker, groups=groups))

        dec_layers = [gInception_ST(channel_hid, channel_hid//2, channel_hid, 
                                   incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(gInception_ST(2*channel_hid, channel_hid//2, channel_hid, 
                                          incep_ker=incep_ker, groups=groups))
        dec_layers.append(gInception_ST(2*channel_hid, channel_hid//2, channel_in, 
                                       incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # Handle temporal flattening with proper channel calculation
        x = x.reshape(B, T*C, H, W)  # Flatten temporal dimension
        
        skips = []
        z = x
        for i, layer in enumerate(self.enc):
            z = layer(z)
            if i < self.N_T - 1:
                skips.append(z)

        z = self.dec[0](z)
        for i in range(1, self.N_T):
            if i < len(skips):
                z = self.dec[i](torch.cat([z, skips[-i]], dim=1))
            else:
                z = self.dec[i](z)

        # Reshape back to original format
        y = z.reshape(B, T, C, H, W)
        return y


class SimVP_Model_GroupNorm(nn.Module):
    """SimVP Model with GroupNorm - fixes BatchNorm channel mismatch"""
    
    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=8, 
                 incep_ker=[3,5,7,11], groups=8, **kwargs):
        super(SimVP_Model_GroupNorm, self).__init__()
        T, C, H, W = in_shape
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, C, N_S)

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid)
        Y = Y.reshape(B, T, C, H, W)
        return Y