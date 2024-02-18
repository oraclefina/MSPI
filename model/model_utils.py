import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import to_2tuple
from einops import rearrange
from backbones.resnet import get_resnet18
from backbones.s3d import *
from fvcore.nn import FlopCountAnalysis, flop_count_table
from timm.models.layers import DropPath, trunc_normal_
from model.get_video_backbones import video_motion_extractor
from timm.models.vision_transformer import VisionTransformer
import numpy as np


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Mlp3d(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Conv3d(in_features, hidden_features, 1, 1, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1, 1, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class SA(nn.Module):
    def __init__(self, in_embed_dim=512, k=2) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=(1, k, k), align_corners=False,
                              mode='trilinear') if k != 1 else nn.Identity()
        self.conv_mask = nn.Sequential(
            BasicConv3d(in_embed_dim, in_embed_dim // 16, kernel_size=3, stride=1, padding=1),
            self.up,
            nn.Conv3d(in_embed_dim // 16, 1, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x, mask):
        mask = self.conv_mask(mask)
        x = x * mask + x
        return x


class Inception(nn.Module):
    def __init__(self, embed_dim=320 + 96):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(embed_dim, 192, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(embed_dim, 96, kernel_size=1, stride=1),
            SepConv3d(96, 208, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(embed_dim, 16, kernel_size=1, stride=1),
            SepConv3d(16, 48, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(embed_dim, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Adapter(nn.Module):
    def __init__(self, embed_dim=320 + 96, num_frames=32, stride=8) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.pool_time = nn.MaxPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1))
        self.conv = Inception(embed_dim=embed_dim)
        self.up = nn.Upsample(scale_factor=(1, 2, 2), align_corners=False, mode='trilinear')

    def forward(self, x):
        o3, o2 = x
        o3 = rearrange(o3, '(b t) c h w -> b c t h w', t=self.num_frames)
        o3_ = self.pool_time(o3)
        o2 = rearrange(o2, '(b t) c h w -> b c t h w', t=self.num_frames)
        o2_ = self.pool_time(o2)
        # print(o2.shape, o3.shape)
        x = torch.concat([o3_, self.up(o2_)], dim=1)
        x = self.conv(x)

        return x


class SyncBlock(nn.Module):
    def __init__(self, num_blocks=3, num_vis_tokens=336, num_aud_tokens=36, vis_in_embed=1024, embed_dim=512) -> None:
        super().__init__()

        self.vis_pos_embed = get_sinusoid_encoding_table(num_vis_tokens, 512)
        self.aud_pos_embed = get_sinusoid_encoding_table(num_aud_tokens, 512)

        self.vis_proj = nn.Linear(vis_in_embed, 512)
        self.vis_norm = nn.LayerNorm(512)

        self.aud_norm = nn.LayerNorm(512)

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=4)
            for _ in range(num_blocks)
        ])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'vis_pos_embed', 'aud_pos_embed'}

    def forward(self, vis_fea, aud_fea):
        # vis_fea: [B, C, T, H, W]
        # aud_fea: [B, C, F, T]
        B, _, t, h, w = vis_fea.shape
        # print(vis_fea.shape, B,self.cls_token.expand(B,-1,-1).shape )
        # print(vis_fea.shape)
        vis_fea = rearrange(vis_fea, 'b c t h w -> b (t h w) c')
        aud_fea = rearrange(aud_fea, 'b c h t -> b (h t) c')

        num_vis_tokens = vis_fea.shape[1]
        # print(vis_fea.shape)

        vis_fea = self.vis_proj(vis_fea)
        vis_fea = self.vis_norm(vis_fea)
        aud_fea = self.aud_norm(aud_fea)
        # print(vis_fea.shape, self.vis_pos_embed.shape)
        vis_fea = vis_fea + self.vis_pos_embed.type_as(vis_fea).to(vis_fea.device).clone().detach()
        aud_fea = aud_fea + self.aud_pos_embed.type_as(aud_fea).to(aud_fea.device).clone().detach()
        # print(vis_fea.shape, aud_fea.shape)

        feas = torch.cat([vis_fea, aud_fea], dim=1)

        for blk in self.blocks:
            feas = blk(feas)

        return feas


def D(p, z, norm=False):
    # src: https://github.com/PatrickHua/SimSiam
    if norm:
        p = F.normalize(p, p=2., dim=-1)
        z = F.normalize(z, p=2., dim=-1)
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()


class LayerNorm3d(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)

        return x


class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        layer_scale_init_value = 1e-6
        self.dwconv_t = nn.Conv3d(dim, dim, kernel_size=(7, 1, 1), padding=(3, 0, 0), groups=dim)  # depthwise conv
        self.dwconv_s = nn.Conv3d(dim, dim, kernel_size=(1, 7, 7), padding=(0, 3, 3), groups=dim)  # depthwise conv
        self.norm = LayerNorm3d(dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, 1)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        # self.grn = GRN(dim*4)
        self.pwconv2 = nn.Conv3d(4 * dim, dim, 1)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1, 1)), 
        #                             requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = self.dwconv_t(x)
        # print(x.shape)
        x = self.dwconv_s(x)
        # print(x.shape)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        # x = self.grn(x)
        x = self.pwconv2(x)
        # if self.gamma is not None:
        #     x = self.gamma * x

        x = input + self.drop_path(x)
        return x


class StaticSaliencyModelConvNext(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = timm.models.create_model("convnext_tiny", pretrained=True, features_only=True)
        embed = 48
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

        self.smooth_0 = nn.Sequential(
            nn.Conv2d(768, 320, 3, 1, 1),
            nn.BatchNorm2d(320),
            nn.ReLU(),
        )

        self.smooth_1 = nn.Sequential(
            nn.Conv2d(384, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )

    def forward(self, x):
        o3, o2, o1, o0 = self.encoder(x)

        o0 = self.smooth_0(o0)
        o1 = self.smooth_1(o1)

        return o1, o0


class AudioVisualSaliencyModel(nn.Module):
    def __init__(self, cfg, vis_embed_dims=(96, 192, 384, 768), aud_embed_dim=512, de_embed_dim=192,
                 num_vis_tokens=4 * 7 * 7, norm=nn.LayerNorm) -> None:
        super().__init__()
        print("Motion Encoder is {}.".format(cfg.MODEL.MOTION_ENCODER))
        self.cfg = cfg
        vis_embed_dims = cfg.MODEL.MOTION_ENCODER_EMBEDS[cfg.MODEL.MOTION_ENCODER]
        num_vis_tokens = cfg.MODEL.NUM_VIS_TOKENS[cfg.MODEL.MOTION_ENCODER]
        self.audnet = get_resnet18(path=cfg.MODEL.AUDIO_ENCODER_WEIGHT)
        self.image_encoder = StaticSaliencyModelConvNext()
        self.visnet = video_motion_extractor(cfg)
        self.aud_vis_sync_block = SyncBlock(num_blocks=3, num_vis_tokens=num_vis_tokens, vis_in_embed=vis_embed_dims[-1],
                                            embed_dim=aud_embed_dim)
        self.aud_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.vis_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        mlp_hidden = 2048
        self.vis_projector = nn.Sequential(
            nn.Linear(aud_embed_dim, mlp_hidden),
            norm(mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, mlp_hidden),
            norm(mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, mlp_hidden),
            norm(mlp_hidden),
        )
        self.mlp_vis = nn.Sequential(
            nn.Linear(mlp_hidden, 512),
            norm(512),
            nn.ReLU(),
            nn.Linear(512, mlp_hidden)
        )
        self.aud_projector = nn.Sequential(
            nn.Linear(aud_embed_dim, mlp_hidden),
            norm(mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, mlp_hidden),
            norm(mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, mlp_hidden),
            norm(mlp_hidden),
        )
        self.mlp_aud = nn.Sequential(
            nn.Linear(mlp_hidden, 512),
            norm(512),
            nn.ReLU(),
            nn.Linear(512, mlp_hidden)
        )

        if cfg.MODEL.LATERAL_BOOL[0]:
            self.latlayer_0 = nn.Sequential(
                nn.Conv3d(vis_embed_dims[0], de_embed_dim, 1, 1, 0),
                nn.Conv3d(de_embed_dim, de_embed_dim, kernel_size=(cfg.MODEL.LATERAL_STRIDE[0], 1, 1),
                          stride=(cfg.MODEL.LATERAL_STRIDE[0], 1, 1), bias=False),
                ConvNextBlock(dim=de_embed_dim),
            )
        else:
            self.latlayer_0 = nn.Sequential(
                nn.Conv3d(vis_embed_dims[0], de_embed_dim, 1, 1, 0),
                ConvNextBlock(dim=de_embed_dim),
            )
        if cfg.MODEL.LATERAL_BOOL[1]:
            self.latlayer_1 = nn.Sequential(
                nn.Conv3d(vis_embed_dims[1], de_embed_dim, 1, 1, 0),
                nn.Conv3d(de_embed_dim, de_embed_dim, kernel_size=(cfg.MODEL.LATERAL_STRIDE[1], 1, 1),
                          stride=(cfg.MODEL.LATERAL_STRIDE[1], 1, 1), bias=False),
                ConvNextBlock(dim=de_embed_dim),
            )
        else:
            self.latlayer_1 = nn.Sequential(
                nn.Conv3d(vis_embed_dims[1], de_embed_dim, 1, 1, 0),
                ConvNextBlock(dim=de_embed_dim),
            )
        if cfg.MODEL.LATERAL_BOOL[2]:
            self.latlayer_2 = nn.Sequential(
                nn.Conv3d(vis_embed_dims[2], de_embed_dim, 1, 1, 0),
                nn.Conv3d(de_embed_dim, de_embed_dim, kernel_size=(cfg.MODEL.LATERAL_STRIDE[2], 1, 1),
                          stride=(cfg.MODEL.LATERAL_STRIDE[2], 1, 1), bias=False),
                ConvNextBlock(dim=de_embed_dim),
            )
        else:
            self.latlayer_2 = nn.Sequential(
                nn.Conv3d(vis_embed_dims[2], de_embed_dim, 1, 1, 0),
                ConvNextBlock(dim=de_embed_dim),
            )
        if cfg.MODEL.LATERAL_BOOL[3]:
            self.latlayer_3 = nn.Sequential(
                nn.Conv3d(vis_embed_dims[3] + aud_embed_dim, de_embed_dim, 1, 1, 0),
                nn.Conv3d(de_embed_dim, de_embed_dim, kernel_size=(cfg.MODEL.LATERAL_STRIDE[3], 1, 1),
                          stride=(cfg.MODEL.LATERAL_STRIDE[3], 1, 1), bias=False),
                ConvNextBlock(dim=de_embed_dim),
            )
        else:
            self.latlayer_3 = nn.Sequential(
                nn.Conv3d(vis_embed_dims[3] + aud_embed_dim, de_embed_dim, 1, 1, 0),
                ConvNextBlock(dim=de_embed_dim),
            )

        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.upsample_4 = nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear', align_corners=False)
        self.upsample_8 = nn.Upsample(scale_factor=(1, 8, 8), mode='trilinear', align_corners=False)

        self.readout = nn.Sequential(
            nn.Conv3d(de_embed_dim * 4, de_embed_dim, 1, 1, 0),
            nn.Conv3d(de_embed_dim, de_embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(de_embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(de_embed_dim, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear', align_corners=False),
            nn.Conv3d(64, 32, kernel_size=(4, 1, 1), stride=(4, 1, 1), padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
        )

        self.adapter = Adapter(num_frames=cfg.DATA.NUM_FRAMES, stride=cfg.DATA.NUM_FRAMES//4)
        self.sa_0 = SA(512, k=4)
        self.sa_1 = SA(512, k=2)
        self.sa_2 = SA(512, k=1)

        # Load Pretrained Weights
        self.visnet.load_weight(cfg.MODEL.MOTION_ENCODER_WEIGHT)
        self.audnet.load_state_dict(torch.load(cfg.MODEL.AUDIO_ENCODER_WEIGHT))
        self.image_encoder.load_state_dict(torch.load(cfg.MODEL.IMAGE_SALIENCY_ENCODER_WEIGHT), strict=False)

    def frozen_encoder(self):
        self.audnet.eval()
        self.image_encoder.eval()

    def forward_encoder(self, clips, audios):
        if self.cfg.MODEL.MOTION_ENCODER == 'slowfast4x16':
            with torch.no_grad():
                clips_s = torch.stack([clips[:,:,0,:,:],clips[:,:,4,:,:], clips[:,:,12,:,:],clips[:,:,-1,:,:]],dim=2)
            clips = [clips_s, clips]
        elif self.cfg.MODEL.MOTION_ENCODER == 's3d':
            clips = clips
        elif 'morph' in self.cfg.MODEL.MOTION_ENCODER:
            clips = clips
        elif 'swin' in self.cfg.MODEL.MOTION_ENCODER:
            clips = clips
        else:
            clips = [clips]
        aud_features = self.audnet(audios)
        v1, v2, v3, v4 = self.visnet(clips)

        # synchronization
        _, _, t, h, w = v4.shape
        _, _, ha, _ = aud_features.shape

        x = self.aud_vis_sync_block(v4, aud_features)
        vis_fea = x[:, :t * h * w, :]
        aud_fea = x[:, t * h * w:, :]
        vis_fea = rearrange(vis_fea, 'b (t h w) c -> b c t h w', t=t, h=h)
        aud_fea = rearrange(aud_fea, 'b (h w) c -> b c h w', h=ha)
        vis_fea_embedding = self.vis_projector(self.vis_pool(vis_fea).flatten(1))
        aud_fea_embedding = self.aud_projector(self.aud_pool(aud_fea).flatten(1))

        vis_pred = self.mlp_vis(vis_fea_embedding)
        aud_pred = self.mlp_aud(aud_fea_embedding)

        va_loss = (D(vis_pred, aud_fea_embedding) + D(aud_pred, vis_fea_embedding)) * 0.5
        loss_va_av = va_loss

        return v1, v2, v3, v4, vis_fea, loss_va_av

    def forward(self, clips, audios):
        masks = self.adapter(self.image_encoder(rearrange(clips, 'b c t h w -> (b t) c h w')))
        v1, v2, v3, v4, vis_sync, loss_av = self.forward_encoder(clips, audios)
        v4 = torch.cat([v4, vis_sync], dim=1)

        s3 = self.latlayer_3(v4)
        s0 = self.latlayer_0(v1)
        s1 = self.latlayer_1(v2)
        s2 = self.latlayer_2(v3)

        s2 = self.sa_2(s2, masks) + self.upsample(s3)
        s1 = self.sa_1(s1, masks) + self.upsample(s2) + self.upsample_4(s3)
        s0 = self.sa_0(s0, masks) + self.upsample(s1) + self.upsample_4(s2) + self.upsample_8(s3)

        out = self.readout(torch.concat([s0, self.upsample(s1), self.upsample_4(s2), self.upsample_8(s3)], dim=1))
        out = out.squeeze(1).squeeze(1)
        out = out - torch.logsumexp(out, dim=(1, 2), keepdim=True)

        return out, loss_av

class VisualSaliencyModel(nn.Module):
    def __init__(self, cfg, vis_embed_dims=(96, 192, 384, 768), aud_embed_dim=512, de_embed_dim=192,
                 num_vis_tokens=4 * 7 * 7, norm=nn.LayerNorm) -> None:
        super().__init__()
        print("Motion Encoder is {}.".format(cfg.MODEL.MOTION_ENCODER))
        self.cfg = cfg
        vis_embed_dims = cfg.MODEL.MOTION_ENCODER_EMBEDS[cfg.MODEL.MOTION_ENCODER]
        num_vis_tokens = cfg.MODEL.NUM_VIS_TOKENS[cfg.MODEL.MOTION_ENCODER]
        self.image_encoder = StaticSaliencyModelConvNext()
        self.visnet = video_motion_extractor(cfg)

        if cfg.MODEL.LATERAL_BOOL[0]:
            self.latlayer_0 = nn.Sequential(
                nn.Conv3d(vis_embed_dims[0], de_embed_dim, 1, 1, 0),
                nn.Conv3d(de_embed_dim, de_embed_dim, kernel_size=(cfg.MODEL.LATERAL_STRIDE[0], 1, 1),
                          stride=(cfg.MODEL.LATERAL_STRIDE[0], 1, 1), bias=False),
                ConvNextBlock(dim=de_embed_dim),
            )
        else:
            self.latlayer_0 = nn.Sequential(
                nn.Conv3d(vis_embed_dims[0], de_embed_dim, 1, 1, 0),
                ConvNextBlock(dim=de_embed_dim),
            )
        if cfg.MODEL.LATERAL_BOOL[1]:
            self.latlayer_1 = nn.Sequential(
                nn.Conv3d(vis_embed_dims[1], de_embed_dim, 1, 1, 0),
                nn.Conv3d(de_embed_dim, de_embed_dim, kernel_size=(cfg.MODEL.LATERAL_STRIDE[1], 1, 1),
                          stride=(cfg.MODEL.LATERAL_STRIDE[1], 1, 1), bias=False),
                ConvNextBlock(dim=de_embed_dim),
            )
        else:
            self.latlayer_1 = nn.Sequential(
                nn.Conv3d(vis_embed_dims[1], de_embed_dim, 1, 1, 0),
                ConvNextBlock(dim=de_embed_dim),
            )
        if cfg.MODEL.LATERAL_BOOL[2]:
            self.latlayer_2 = nn.Sequential(
                nn.Conv3d(vis_embed_dims[2], de_embed_dim, 1, 1, 0),
                nn.Conv3d(de_embed_dim, de_embed_dim, kernel_size=(cfg.MODEL.LATERAL_STRIDE[2], 1, 1),
                          stride=(cfg.MODEL.LATERAL_STRIDE[2], 1, 1), bias=False),
                ConvNextBlock(dim=de_embed_dim),
            )
        else:
            self.latlayer_2 = nn.Sequential(
                nn.Conv3d(vis_embed_dims[2], de_embed_dim, 1, 1, 0),
                ConvNextBlock(dim=de_embed_dim),
            )
        if cfg.MODEL.LATERAL_BOOL[3]:
            self.latlayer_3 = nn.Sequential(
                nn.Conv3d(vis_embed_dims[3] , de_embed_dim, 1, 1, 0),
                nn.Conv3d(de_embed_dim, de_embed_dim, kernel_size=(cfg.MODEL.LATERAL_STRIDE[3], 1, 1),
                          stride=(cfg.MODEL.LATERAL_STRIDE[3], 1, 1), bias=False),
                ConvNextBlock(dim=de_embed_dim),
            )
        else:
            self.latlayer_3 = nn.Sequential(
                nn.Conv3d(vis_embed_dims[3] , de_embed_dim, 1, 1, 0),
                ConvNextBlock(dim=de_embed_dim),
            )

        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.upsample_4 = nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear', align_corners=False)
        self.upsample_8 = nn.Upsample(scale_factor=(1, 8, 8), mode='trilinear', align_corners=False)

        self.readout = nn.Sequential(
            nn.Conv3d(de_embed_dim * 4, de_embed_dim, 1, 1, 0),
            nn.Conv3d(de_embed_dim, de_embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(de_embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(de_embed_dim, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear', align_corners=False),
            nn.Conv3d(64, 32, kernel_size=(4, 1, 1), stride=(4, 1, 1), padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
        )

        self.adapter = Adapter(num_frames=cfg.DATA.NUM_FRAMES, stride=cfg.DATA.NUM_FRAMES//4)
        self.sa_0 = SA(512, k=4)
        self.sa_1 = SA(512, k=2)
        self.sa_2 = SA(512, k=1)

        # Load Pretrained Weights
        self.visnet.load_weight(cfg.MODEL.MOTION_ENCODER_WEIGHT)
        self.image_encoder.load_state_dict(torch.load(cfg.MODEL.IMAGE_SALIENCY_ENCODER_WEIGHT), strict=False)

    def frozen_encoder(self):
        self.image_encoder.eval()

    def forward_encoder(self, clips):
        if self.cfg.MODEL.MOTION_ENCODER == 'slowfast4x16':
            with torch.no_grad():
                clips_s = torch.stack([clips[:,:,0,:,:],clips[:,:,4,:,:], clips[:,:,12,:,:],clips[:,:,-1,:,:]],dim=2)
            clips = [clips_s, clips]
        elif self.cfg.MODEL.MOTION_ENCODER == 's3d':
            clips = clips
        elif 'morph' in self.cfg.MODEL.MOTION_ENCODER:
            clips = clips
        elif 'swin' in self.cfg.MODEL.MOTION_ENCODER:
            clips = clips
        else:
            clips = [clips]
        v1, v2, v3, v4 = self.visnet(clips)

        return v1, v2, v3, v4

    def forward(self, clips):
        masks = self.adapter(self.image_encoder(rearrange(clips, 'b c t h w -> (b t) c h w')))
        v1, v2, v3, v4 = self.forward_encoder(clips)

        s3 = self.latlayer_3(v4)
        s0 = self.latlayer_0(v1)
        s1 = self.latlayer_1(v2)
        s2 = self.latlayer_2(v3)

        s2 = self.sa_2(s2, masks) + self.upsample(s3)
        s1 = self.sa_1(s1, masks) + self.upsample(s2) + self.upsample_4(s3)
        s0 = self.sa_0(s0, masks) + self.upsample(s1) + self.upsample_4(s2) + self.upsample_8(s3)

        out = self.readout(torch.concat([s0, self.upsample(s1), self.upsample_4(s2), self.upsample_8(s3)], dim=1))
        out = out.squeeze(1).squeeze(1)
        out = out - torch.logsumexp(out, dim=(1, 2), keepdim=True)

        return out, 0

from config import cfg
if __name__ == "__main__":
    model = AudioVisualSaliencyModel(cfg=cfg)
    model.eval()
    fca = FlopCountAnalysis(model, (torch.randn(1, 3, 16, 224, 224), torch.randn(1, 1, 257, 111)))

    print(flop_count_table(fca))
