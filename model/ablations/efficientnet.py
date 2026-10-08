"""Preserved EfficientNet-B4 experiment from myUniAD_Eff.py.

This historical variant also changes decoder depth and uses NonLocalBlock.
It has not been verified as a backbone-only ECFR ablation.
Its original checkpoint path and parameter attributes are retained.
"""
import torch
import torch.nn as nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from timm.models.resnet import Bottleneck

from model import get_model
from model import MODEL

import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import timm
from timm.models.resnet import _cfg


class NonLocalBlock(nn.Module):
    """ NonLocalBlock Module"""

    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()

        conv_nd = nn.Conv2d

        self.in_channels = in_channels
        self.inter_channels = self.in_channels // 2

        self.ImageAfterASPP_bnRelu = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.DepthAfterASPP_bnRelu = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.R_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.R_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.R_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.R_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)

        self.F_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.F_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.F_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.F_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, self_fea, mutual_fea, alpha, selfImage):

        if selfImage:
            selfNonLocal_fea = self.ImageAfterASPP_bnRelu(self_fea)
            mutualNonLocal_fea = self.DepthAfterASPP_bnRelu(mutual_fea)

            batch_size = selfNonLocal_fea.size(0)

            g_x = self.R_g(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            theta_x = self.F_theta(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.F_phi(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)

            self_theta_x = self.R_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_theta_x = self_theta_x.permute(0, 2, 1)
            self_phi_x = self.R_phi(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_f = torch.matmul(self_theta_x, self_phi_x)

            f_div_C = F.softmax(alpha * f + self_f, dim=-1)

            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
            W_y = self.R_W(y)
            z = W_y + self_fea
            return z

        else:
            selfNonLocal_fea = self.DepthAfterASPP_bnRelu(self_fea)
            mutualNonLocal_fea = self.ImageAfterASPP_bnRelu(mutual_fea)

            batch_size = selfNonLocal_fea.size(0)

            g_x = self.F_g(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            theta_x = self.R_theta(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.R_phi(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)

            self_theta_x = self.F_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_theta_x = self_theta_x.permute(0, 2, 1)
            self_phi_x = self.F_phi(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_f = torch.matmul(self_theta_x, self_phi_x)

            f_div_C = F.softmax(alpha * f + self_f, dim=-1)

            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
            W_y = self.F_W(y)
            z = W_y + self_fea
            return z


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv2x2(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, groups=groups, bias=False,
                              dilation=dilation)


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 agent_num=49, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0], 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1]))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.agent_tokens = nn.Parameter(torch.zeros(1, pool_size, pool_size, self.dim))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        b, h, w, c = x.shape
        n = h * w
        x = x.reshape(b, -1, c)
        y = y.reshape(b, -1, c)
        num_heads = self.num_heads
        head_dim = c // num_heads
        q = self.q(x)
        k = self.k(y)
        v = self.v(x)

        agent_tokens = self.agent_tokens.repeat(b, 1, 1, 1).reshape(b, -1, c)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        kv_size = (self.window_size[0], self.window_size[1])
        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x).contiguous().view(b, h, w, -1)
        return x


class AttentionBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop: float = 0,
            proj_drop: float = 0.,
            agent_num: int = 49,
            num_heads: int = 8,
            num_patches: int = 49,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = CrossAttention(dim=hidden_dim, num_patches=num_patches, num_heads=num_heads,
                                                  attn_drop=attn_drop,
                                                  proj_drop=proj_drop, agent_num=agent_num, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor, cross_input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input), self.ln_1(cross_input)))
        return x


class HAC(nn.Module):
    def __init__(
            self,
            depth: int = 2,
            hidden_dim: int = 0,
            drop_path: float = 0,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            agent_num: int = 49,
            num_heads: int = 8,
            num_patches: int = 49,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
    ):
        super().__init__()
        self.attn_blocks = nn.ModuleList([
            AttentionBlock(hidden_dim=hidden_dim, num_patches=num_patches, num_heads=num_heads, attn_drop=attn_drop,
                     proj_drop=proj_drop, agent_num=agent_num, norm_layer=norm_layer, drop_path=drop_path)
            for i in range(depth)])
        self.conv1b3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.conv1a3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.conv1b5 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.conv1a5 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.conv33 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1, bias=False,
                      groups=hidden_dim),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.conv55 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, stride=1, padding=2, bias=False,
                      groups=hidden_dim),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.conv77 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, bias=False,
                      groups=hidden_dim),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.finalconv11 = nn.Conv2d(in_channels=hidden_dim * 3, out_channels=hidden_dim, kernel_size=1, stride=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
		initialization
		"""
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, input: torch.Tensor, cross_input: torch.Tensor):
        for i, blk in enumerate(self.attn_blocks):
            if i == 0:
                out_attn = blk(input, cross_input)
            else:
                out_attn = blk(out_attn, out_attn)
        input_conv = input.permute(0, 3, 1, 2).contiguous()
        out_77 = self.conv1a3(self.conv77(self.conv1b3(input_conv)))
        out_55 = self.conv1a5(self.conv55(self.conv1b5(input_conv)))
        output = torch.cat((out_attn.permute(0, 3, 1, 2).contiguous(), out_55, out_77), dim=1)
        output = self.finalconv11(output).permute(0, 2, 3, 1).contiguous()
        return output + input


class DecoderStage(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            upsample=None,
            depth=2,
            drop_path=0,
            attn_drop=0.,
            proj_drop=0.,
            agent_num=49,
            num_heads=8,
            num_patches=49,
            norm_layer=nn.LayerNorm,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        if depth % 3 == 0:
            self.blocks = nn.ModuleList([
                HAC(hidden_dim=dim, depth=depth,
                              drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                              attn_drop=attn_drop, proj_drop=proj_drop, agent_num=agent_num, num_heads=num_heads,
                              num_patches=num_patches)
                for i in range(depth // 3)])
        elif depth % 2 == 0:
            self.blocks = nn.ModuleList([
                HAC(hidden_dim=dim, depth=depth,
                              drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                              attn_drop=attn_drop, proj_drop=proj_drop, agent_num=agent_num, num_heads=num_heads,
                              num_patches=num_patches)
                for i in range(depth // 2)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, input, cross_input):
        if self.upsample is not None:
            input = self.upsample(input)
            cross_input = self.upsample(cross_input)
        for i, blk in enumerate(self.blocks):
            if i == 0:
                output = blk(input, cross_input)
            else:
                output = blk(output, output)
        return output


class Decoder(nn.Module):
    def __init__(self, dims_decoder=[512, 256, 128, 64], depths_decoder=[3, 4, 6, 3], attn_drop=0., proj_drop=0.,
                 agent_num=[49, 49, 16, 9], drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, num_heads=[8, 4, 2, 1], num_patches=[8 * 8, 16 * 16, 32 * 32, 64 * 64], ):
        super().__init__()
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]
        self.rgb_layers_up = nn.ModuleList()
        self.depth_layers_up = nn.ModuleList()
        self.affinityAttConv_up = nn.ModuleList()
        self.NonLocal_up = nn.ModuleList()
        self.upsample_up = nn.ModuleList()
        self.image_bn_relu_up = nn.ModuleList()
        self.depth_bn_relu_up = nn.ModuleList()
        for i_layer in range(len(depths_decoder)):
            rgb_layer = DecoderStage(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                agent_num=agent_num[i_layer],
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=None,
                num_heads=num_heads[i_layer],
                num_patches=num_patches[i_layer],
            )
            depth_layer = DecoderStage(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                agent_num=agent_num[i_layer],
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=None,
                num_heads=num_heads[i_layer],
                num_patches=num_patches[i_layer],
            )
            affinityAttConv = nn.Sequential(
                nn.Conv2d(in_channels=dims_decoder[i_layer] * 2, out_channels=2, kernel_size=1),
                nn.BatchNorm2d(2),
                nn.ReLU(inplace=True),
            )

            NonLocal = NonLocalBlock(in_channels=dims_decoder[i_layer])
            upsample = PatchExpand2D(dims_decoder[i_layer], norm_layer=norm_layer)
            image_bn_relu = nn.Sequential(
                nn.BatchNorm2d(dims_decoder[i_layer]),
                nn.ReLU(inplace=True))
            depth_bn_relu = nn.Sequential(
                nn.BatchNorm2d(dims_decoder[i_layer]),
                nn.ReLU(inplace=True))
            self.upsample_up.append(upsample)
            self.rgb_layers_up.append(rgb_layer)
            self.depth_layers_up.append(depth_layer)
            self.affinityAttConv_up.append(affinityAttConv)
            self.NonLocal_up.append(NonLocal)
            self.image_bn_relu_up.append(image_bn_relu)
            self.depth_bn_relu_up.append(depth_bn_relu)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, rgb, depth):
        rgb = rearrange(rgb, 'b c h w -> b h w c')
        depth = rearrange(depth, 'b c h w -> b h w c')
        rgb_out_features = []
        depth_out_features = []
        for i, (rgb_layer, depth_layer, affinityAttConv, NonLocal, upsample, image_bn_relu, depth_bn_relu) in \
                enumerate(zip(self.rgb_layers_up, self.depth_layers_up, self.affinityAttConv_up, self.NonLocal_up,
                              self.upsample_up, self.image_bn_relu_up, self.depth_bn_relu_up)):
            if i != 0:
                rgb = upsample(rgb)
                depth = upsample(depth)

            rgb = rearrange(rgb, 'b h w c -> b c h w')
            depth = rearrange(depth, 'b h w c -> b c h w')
            b, c, h, w = rgb.shape
            affinityAtt = F.softmax(affinityAttConv(torch.cat([rgb, depth], dim=1)))
            alphaD = affinityAtt[:, 0, :, :].reshape([b, h * w, 1])
            alphaR = affinityAtt[:, 1, :, :].reshape([b, h * w, 1])

            alphaD = alphaD.expand([b, h * w, h * w])
            alphaR = alphaR.expand([b, h * w, h * w])

            ImageAfterAtt1 = NonLocal(rgb, depth, alphaD, selfImage=True)
            DepthAfterAtt1 = NonLocal(rgb, depth, alphaR, selfImage=False)

            ImageAfterAtt = image_bn_relu(ImageAfterAtt1)
            DepthAfterAtt = depth_bn_relu(DepthAfterAtt1)

            rgb = rearrange(rgb, 'b c h w -> b h w c')
            depth = rearrange(depth, 'b c h w -> b h w c')
            ImageAfterAtt = rearrange(ImageAfterAtt, 'b c h w -> b h w c')
            DepthAfterAtt = rearrange(DepthAfterAtt, 'b c h w -> b h w c')
            rgb = rgb_layer(rgb, ImageAfterAtt)
            depth = depth_layer(depth, DepthAfterAtt)

            if i != 0:
                rgb_out_features.insert(0, rearrange(rgb, 'b h w c -> b c h w'))
                depth_out_features.insert(0, rearrange(depth, 'b h w c -> b c h w'))
        return rgb_out_features, depth_out_features


class FeatureAggregation(nn.Module):
    def __init__(self, block, layers, width_per_group=64, norm_layer=None, ):
        super(FeatureAggregation, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.base_width = width_per_group
        print(block.expansion)
        self.inplanes = 64 * block.expansion
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 128, layers, stride=2)

        self.conv1 = conv3x3(16 * block.expansion, 32 * block.expansion, 2)
        self.bn1 = norm_layer(32 * block.expansion)
        self.conv2 = conv3x3(32 * block.expansion, 64 * block.expansion, 2)
        self.bn2 = norm_layer(64 * block.expansion)
        self.conv21 = nn.Conv2d(32 * block.expansion, 32 * block.expansion, 1)
        self.bn21 = norm_layer(32 * block.expansion)
        self.conv31 = nn.Conv2d(64 * block.expansion, 64 * block.expansion, 1)
        self.bn31 = norm_layer(64 * block.expansion)
        self.convf = nn.Conv2d(64 * block.expansion, 64 * block.expansion, 1)
        self.bnf = norm_layer(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.conv_layer1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1)
        self.conv_layer2 = nn.Conv2d(in_channels=56, out_channels=128, kernel_size=1, stride=1)
        self.conv_layer3 = nn.Conv2d(in_channels=160, out_channels=256, kernel_size=1, stride=1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion), )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, base_width=self.base_width, dilation=previous_dilation,
                  norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        fpn0 = self.relu(self.bn1(self.conv1(x[0])))
        fpn1 = self.relu(self.bn21(self.conv21(x[1]))) + fpn0
        sv_features = self.relu(self.bn2(self.conv2(fpn1))) + self.relu(self.bn31(self.conv31(x[2])))
        sv_features = self.relu(self.bnf(self.convf(sv_features)))
        sv_features = self.bn_layer(sv_features)

        return sv_features.contiguous()


class EfficientNetVariant(nn.Module):
    def __init__(self, model_t, model_s):
        super(EfficientNetVariant, self).__init__()
        config = _cfg(url='', file='../backbone_checkpoint/efficientnet_b4_weights.pth')
        self.net_t = timm.create_model('efficientnet_b4', pretrained=True, features_only=True, out_indices=[1, 2, 3],
                                       pretrained_cfg=config)
        self.rgb_mff_oce = FeatureAggregation(Bottleneck, 3)
        self.depth_mff_oce = FeatureAggregation(Bottleneck, 3)
        self.net_s = Decoder()

        self.frozen_layers = ['net_t']

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def forward(self, imgs, depth):
        rgb_feats_t = self.net_t(imgs)
        depth_feats_t = self.net_t(depth)
        rgb_feats_t = [f.detach() for f in rgb_feats_t]
        depth_feats_t = [f.detach() for f in depth_feats_t]
        rgb_feats_s, depth_feats_s = self.net_s(self.rgb_mff_oce(rgb_feats_t), self.depth_mff_oce(depth_feats_t))
        return rgb_feats_t, depth_feats_t, rgb_feats_s, depth_feats_s


@MODEL.register_module
def ecfr_efficientnet(pretrained=False, **kwargs):
    model = EfficientNetVariant(**kwargs)
    return model
