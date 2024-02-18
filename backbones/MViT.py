import sys
import torch
import torch.nn as nn
import math
from functools import partial
import SlowFast.stem_helper as stem_helper
import SlowFast.resnet_helper as resnet_helper
from SlowFast.slowfast.config.defaults import assert_and_infer_cfg
from SlowFast.slowfast.utils.parser import load_config, parse_args
from fvcore.nn import FlopCountAnalysis, flop_count_table
from SlowFast.slowfast.utils.checkpoint import load_checkpoint
import numpy as np
import numpy
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
from torch.autograd import Function as Function
from einops import rearrange


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop_rate=0.0,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        if self.drop_rate > 0.0:
            self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        return x


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class TwoStreamFusion(nn.Module):
    def __init__(self, mode, dim=None, kernel=3, padding=1):
        """
        A general constructor for neural modules fusing two equal sized tensors
        in forward. Following options are supported:

        "add" / "max" / "min" / "avg"             : respective operations on the two halves.
        "concat"                                  : NOOP.
        "concat_linear_{dim_mult}_{drop_rate}"    : MLP to fuse with hidden dim "dim_mult"
                                                    (optional, def 1.) higher than input dim
                                                    with optional dropout "drop_rate" (def: 0.)
        "ln+concat_linear_{dim_mult}_{drop_rate}" : perform MLP after layernorm on the input.

        """
        super().__init__()
        self.mode = mode
        if mode == "add":
            self.fuse_fn = lambda x: torch.stack(torch.chunk(x, 2, dim=2)).sum(
                dim=0
            )
        elif mode == "max":
            self.fuse_fn = (
                lambda x: torch.stack(torch.chunk(x, 2, dim=2))
                .max(dim=0)
                .values
            )
        elif mode == "min":
            self.fuse_fn = (
                lambda x: torch.stack(torch.chunk(x, 2, dim=2))
                .min(dim=0)
                .values
            )
        elif mode == "avg":
            self.fuse_fn = lambda x: torch.stack(torch.chunk(x, 2, dim=2)).mean(
                dim=0
            )
        elif mode == "concat":
            # x itself is the channel concat version
            self.fuse_fn = lambda x: x
        elif "concat_linear" in mode:
            if len(mode.split("_")) == 2:
                dim_mult = 1.0
                drop_rate = 0.0
            elif len(mode.split("_")) == 3:
                dim_mult = float(mode.split("_")[-1])
                drop_rate = 0.0

            elif len(mode.split("_")) == 4:
                dim_mult = float(mode.split("_")[-2])
                drop_rate = float(mode.split("_")[-1])
            else:
                raise NotImplementedError

            if mode.split("+")[0] == "ln":
                self.fuse_fn = nn.Sequential(
                    nn.LayerNorm(dim),
                    Mlp(
                        in_features=dim,
                        hidden_features=int(dim * dim_mult),
                        act_layer=nn.GELU,
                        out_features=dim,
                        drop_rate=drop_rate,
                    ),
                )
            else:
                self.fuse_fn = Mlp(
                    in_features=dim,
                    hidden_features=int(dim * dim_mult),
                    act_layer=nn.GELU,
                    out_features=dim,
                    drop_rate=drop_rate,
                )

        else:
            raise NotImplementedError

    def forward(self, x):
        if "concat_linear" in self.mode:
            return self.fuse_fn(x) + x

        else:
            return self.fuse_fn(x)


def attention_pool(tensor, pool, thw_shape, has_cls_embed=True, norm=None):
    if pool is None:
        return tensor, thw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = (
        tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    )

    tensor = pool(tensor)

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)
    # Assert tensor_dim in [3, 4]
    if tensor_dim == 4:
        pass
    else:  # tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, thw_shape


def get_rel_pos(rel_pos, d):
    if isinstance(d, int):
        ori_d = rel_pos.shape[0]
        if ori_d == d:
            return rel_pos
        else:
            # Interpolate rel pos.
            new_pos_embed = F.interpolate(
                rel_pos.reshape(1, ori_d, -1).permute(0, 2, 1),
                size=d,
                mode="linear",
            )

            return new_pos_embed.reshape(-1, d).permute(1, 0)


class ReversibleMViT(nn.Module):
    """
    Reversible model builder. This builds the reversible transformer encoder
    and allows reversible training.

    Karttikeya Mangalam, Haoqi Fan, Yanghao Li, Chao-Yuan Wu, Bo Xiong,
    Christoph Feichtenhofer, Jitendra Malik
    "Reversible Vision Transformers"

    https://openaccess.thecvf.com/content/CVPR2022/papers/Mangalam_Reversible_Vision_Transformers_CVPR_2022_paper.pdf
    """

    def __init__(self, config, model):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
            model (nn.Module): parent MViT module this module forms
                a reversible encoder in.
        """

        super().__init__()
        self.cfg = config

        embed_dim = self.cfg.MVIT.EMBED_DIM
        depth = self.cfg.MVIT.DEPTH
        num_heads = self.cfg.MVIT.NUM_HEADS
        mlp_ratio = self.cfg.MVIT.MLP_RATIO
        qkv_bias = self.cfg.MVIT.QKV_BIAS

        drop_path_rate = self.cfg.MVIT.DROPPATH_RATE
        self.dropout = config.MVIT.DROPOUT_RATE
        self.pre_q_fusion = self.cfg.MVIT.REV.PRE_Q_FUSION
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        input_size = model.patch_dims

        self.layers = nn.ModuleList([])
        self.no_custom_backward = False

        if self.cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(self.cfg.MVIT.DIM_MUL)):
            dim_mul[self.cfg.MVIT.DIM_MUL[i][0]] = self.cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(self.cfg.MVIT.HEAD_MUL)):
            head_mul[self.cfg.MVIT.HEAD_MUL[i][0]] = self.cfg.MVIT.HEAD_MUL[i][
                1
            ]

        pool_q = model.pool_q
        pool_kv = model.pool_kv
        stride_q = model.stride_q
        stride_kv = model.stride_kv

        for i in range(depth):

            num_heads = round_width(num_heads, head_mul[i])

            # Upsampling inside the MHPA, input to the Q-pooling block is lower C dimension
            # This localizes the feature changes in a single block, making more computation reversible.
            embed_dim = round_width(
                embed_dim, dim_mul[i - 1] if i > 0 else 1.0, divisor=num_heads
            )
            dim_out = round_width(
                embed_dim,
                dim_mul[i],
                divisor=round_width(num_heads, head_mul[i + 1]),
            )

            if i in self.cfg.MVIT.REV.BUFFER_LAYERS:
                layer_type = StageTransitionBlock
                input_mult = 2 if "concat" in self.pre_q_fusion else 1
            else:
                layer_type = ReversibleBlock
                input_mult = 1

            dimout_correction = (
                2 if (input_mult == 2 and "concat" in self.pre_q_fusion) else 1
            )

            self.layers.append(
                layer_type(
                    dim=embed_dim
                        * input_mult,  # added only for concat fusion before Qpooling layers
                    input_size=input_size,
                    dim_out=dim_out * input_mult // dimout_correction,
                    num_heads=num_heads,
                    cfg=self.cfg,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    kernel_q=pool_q[i] if len(pool_q) > i else [],
                    kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                    stride_q=stride_q[i] if len(stride_q) > i else [],
                    stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                    layer_id=i,
                    pre_q_fusion=self.pre_q_fusion,
                )
            )
            # F is the attention block
            self.layers[-1].F.thw = input_size

            if len(stride_q[i]) > 0:
                input_size = [
                    size // stride
                    for size, stride in zip(input_size, stride_q[i])
                ]

        embed_dim = dim_out

    @staticmethod
    def vanilla_backward(h, layers, buffer):
        """
        Using rev layers without rev backpropagation. Debugging purposes only.
        Activated with self.no_custom_backward.
        """

        # split into hidden states (h) and attention_output (a)
        h, a = torch.chunk(h, 2, dim=-1)
        for _, layer in enumerate(layers):
            a, h = layer(a, h)

        return torch.cat([a, h], dim=-1)

    def forward(self, x):

        # process the layers in a reversible stack and an irreversible stack.
        stack = []
        for l_i in range(len(self.layers)):
            if isinstance(self.layers[l_i], StageTransitionBlock):
                stack.append(("StageTransition", l_i))
            else:
                if len(stack) == 0 or stack[-1][0] == "StageTransition":
                    stack.append(("Reversible", []))
                stack[-1][1].append(l_i)

        for layer_seq in stack:

            if layer_seq[0] == "StageTransition":
                x = self.layers[layer_seq[1]](x)

            else:
                x = torch.cat([x, x], dim=-1)

                # no need for custom backprop in eval/model stat log
                if not self.training or self.no_custom_backward:
                    executing_fn = ReversibleMViT.vanilla_backward
                else:
                    executing_fn = RevBackProp.apply

                x = executing_fn(
                    x,
                    self.layers[layer_seq[1][0]: layer_seq[1][-1] + 1],
                    [],  # buffer activations
                )

        # Apply dropout
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        return x


class RevBackProp(Function):
    """
    Custom Backpropagation function to allow (A) flusing memory in foward
    and (B) activation recomputation reversibly in backward for gradient calculation.

    Inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
    """

    @staticmethod
    def forward(
            ctx,
            x,
            layers,
            buffer_layers,  # List of layer ids for int activation to buffer
    ):
        """
        Reversible Forward pass. Any intermediate activations from `buffer_layers` are
        cached in ctx for forward pass. This is not necessary for standard usecases.
        Each reversible layer implements its own forward pass logic.
        """
        buffer_layers.sort()

        X_1, X_2 = torch.chunk(x, 2, dim=-1)

        intermediate = []

        for layer in layers:

            X_1, X_2 = layer(X_1, X_2)

            if layer.layer_id in buffer_layers:
                intermediate.extend([X_1.detach(), X_2.detach()])

        if len(buffer_layers) == 0:
            all_tensors = [X_1.detach(), X_2.detach()]
        else:
            intermediate = [torch.LongTensor(buffer_layers), *intermediate]
            all_tensors = [X_1.detach(), X_2.detach(), *intermediate]

        ctx.save_for_backward(*all_tensors)
        ctx.layers = layers

        return torch.cat([X_1, X_2], dim=-1)

    @staticmethod
    def backward(ctx, dx):
        """
        Reversible Backward pass. Any intermediate activations from `buffer_layers` are
        recovered from ctx. Each layer implements its own loic for backward pass (both
        activation recomputation and grad calculation).
        """
        dX_1, dX_2 = torch.chunk(dx, 2, dim=-1)

        # retrieve params from ctx for backward
        X_1, X_2, *int_tensors = ctx.saved_tensors

        # no buffering
        if len(int_tensors) != 0:
            buffer_layers = int_tensors[0].tolist()

        else:
            buffer_layers = []

        layers = ctx.layers

        for _, layer in enumerate(layers[::-1]):

            if layer.layer_id in buffer_layers:

                X_1, X_2, dX_1, dX_2 = layer.backward_pass(
                    Y_1=int_tensors[
                        buffer_layers.index(layer.layer_id) * 2 + 1
                        ],
                    Y_2=int_tensors[
                        buffer_layers.index(layer.layer_id) * 2 + 2
                        ],
                    dY_1=dX_1,
                    dY_2=dX_2,
                )

            else:

                X_1, X_2, dX_1, dX_2 = layer.backward_pass(
                    Y_1=X_1,
                    Y_2=X_2,
                    dY_1=dX_1,
                    dY_2=dX_2,
                )

        dx = torch.cat([dX_1, dX_2], dim=-1)

        del int_tensors
        del dX_1, dX_2, X_1, X_2

        return dx, None, None


class StageTransitionBlock(nn.Module):
    """
    Blocks for changing the feature dimensions in MViT (using Q-pooling).
    See Section 3.3.1 in paper for details.
    """

    def __init__(
            self,
            dim,
            input_size,
            dim_out,
            num_heads,
            mlp_ratio,
            qkv_bias,
            drop_path,
            kernel_q,
            kernel_kv,
            stride_q,
            stride_kv,
            cfg,
            norm_layer=nn.LayerNorm,
            pre_q_fusion=None,
            layer_id=0,
    ):
        """
        Uses the same structure of F and G functions as Reversible Block except
        without using reversible forward (and backward) pass.
        """
        super().__init__()

        self.drop_path_rate = drop_path

        embed_dim = dim

        self.F = AttentionSubBlock(
            dim=embed_dim,
            input_size=input_size,
            num_heads=num_heads,
            cfg=cfg,
            dim_out=dim_out,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
        )

        self.G = MLPSubblock(
            dim=dim_out,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )

        self.layer_id = layer_id

        self.is_proj = False
        self.has_cls_embed = cfg.MVIT.CLS_EMBED_ON

        self.is_conv = False
        self.pool_first = cfg.MVIT.POOL_FIRST
        self.mode = cfg.MVIT.MODE
        self.pre_q_fuse = TwoStreamFusion(pre_q_fusion, dim=dim)

        if cfg.MVIT.REV.RES_PATH == "max":
            self.res_conv = False
            self.pool_skip = nn.MaxPool3d(
                # self.attention.attn.pool_q.kernel_size,
                [s + 1 if s > 1 else s for s in self.F.attn.pool_q.stride],
                self.F.attn.pool_q.stride,
                [int(k // 2) for k in self.F.attn.pool_q.stride],
                # self.attention.attn.pool_q.padding,
                ceil_mode=False,
            )

        elif cfg.MVIT.REV.RES_PATH == "conv":
            self.res_conv = True
        else:
            raise NotImplementedError

        # Add a linear projection in residual branch
        if embed_dim != dim_out:
            self.is_proj = True
            self.res_proj = nn.Linear(embed_dim, dim_out, bias=True)

    def forward(
            self,
            x,
    ):
        """
        Forward logic is similar to MultiScaleBlock with Q-pooling.
        """
        x = self.pre_q_fuse(x)

        # fork tensor for residual connections
        x_res = x

        # This uses conv to pool the residual hidden features
        # but done before pooling only if not pool_first
        if self.is_proj and not self.pool_first:
            x_res = self.res_proj(x_res)

        if self.res_conv:

            # Pooling the hidden features with the same conv as Q
            N, L, C = x_res.shape

            # This handling is the same as that of q in MultiScaleAttention
            if self.mode == "conv_unshared":
                fold_dim = 1
            else:
                fold_dim = self.F.attn.num_heads

            # Output is (B, N, L, C)
            x_res = x_res.reshape(N, L, fold_dim, C // fold_dim).permute(
                0, 2, 1, 3
            )

            x_res, _ = attention_pool(
                x_res,
                self.F.attn.pool_q,
                # thw_shape = self.attention.attn.thw,
                thw_shape=self.F.thw,
                has_cls_embed=self.has_cls_embed,
                norm=self.F.attn.norm_q
                if hasattr(self.F.attn, "norm_q")
                else None,
            )
            x_res = x_res.permute(0, 2, 1, 3).reshape(N, x_res.shape[2], C)

        else:
            # Pooling the hidden features with max op
            x_res, _ = attention_pool(
                x_res,
                self.pool_skip,
                thw_shape=self.F.attn.thw,
                has_cls_embed=self.has_cls_embed,
            )

        # If pool_first then project to higher dim now
        if self.is_proj and self.pool_first:
            x_res = self.res_proj(x_res)

        x = self.F(x)
        x = x_res + x
        x = x + self.G(x)

        x = drop_path(x, drop_prob=self.drop_path_rate, training=self.training)

        return x


class ReversibleBlock(nn.Module):
    """
    Reversible Blocks for Reversible Vision Transformer and also
    for state-preserving blocks in Reversible MViT. See Section
    3.3.2 in paper for details.
    """

    def __init__(
            self,
            dim,
            input_size,
            dim_out,
            num_heads,
            mlp_ratio,
            qkv_bias,
            drop_path,
            kernel_q,
            kernel_kv,
            stride_q,
            stride_kv,
            cfg,
            norm_layer=nn.LayerNorm,
            layer_id=0,
            **kwargs
    ):
        """
        Block is composed entirely of function F (Attention
        sub-block) and G (MLP sub-block) including layernorm.
        """
        super().__init__()

        self.drop_path_rate = drop_path

        self.F = AttentionSubBlock(
            dim=dim,
            input_size=input_size,
            num_heads=num_heads,
            cfg=cfg,
            dim_out=dim_out,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
        )

        self.G = MLPSubblock(
            dim=dim,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )

        self.layer_id = layer_id

        self.seeds = {}

    def seed_cuda(self, key):
        """
        Fix seeds to allow for stochastic elements such as
        dropout to be reproduced exactly in activation
        recomputation in the backward pass.
        """

        # randomize seeds
        # use cuda generator if available
        if (
                hasattr(torch.cuda, "default_generators")
                and len(torch.cuda.default_generators) > 0
        ):
            # GPU
            device_idx = torch.cuda.current_device()
            seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            seed = int(torch.seed() % sys.maxsize)

        self.seeds[key] = seed
        torch.manual_seed(self.seeds[key])

    def forward(self, X_1, X_2):
        """
        forward pass equations:
        Y_1 = X_1 + Attention(X_2), F = Attention
        Y_2 = X_2 + MLP(Y_1), G = MLP
        """

        self.seed_cuda("attn")
        # Y_1 : attn_output
        f_X_2 = self.F(X_2)

        self.seed_cuda("droppath")
        f_X_2_dropped = drop_path(
            f_X_2, drop_prob=self.drop_path_rate, training=self.training
        )

        # Y_1 = X_1 + f(X_2)
        Y_1 = X_1 + f_X_2_dropped

        # free memory
        del X_1

        self.seed_cuda("FFN")
        g_Y_1 = self.G(Y_1)

        torch.manual_seed(self.seeds["droppath"])
        g_Y_1_dropped = drop_path(
            g_Y_1, drop_prob=self.drop_path_rate, training=self.training
        )

        # Y_2 = X_2 + g(Y_1)
        Y_2 = X_2 + g_Y_1_dropped

        del X_2

        return Y_1, Y_2

    def backward_pass(
            self,
            Y_1,
            Y_2,
            dY_1,
            dY_2,
    ):
        """
        equation for activation recomputation:
        X_2 = Y_2 - G(Y_1), G = MLP
        X_1 = Y_1 - F(X_2), F = Attention
        """

        # temporarily record intermediate activation for G
        # and use them for gradient calculcation of G
        with torch.enable_grad():
            Y_1.requires_grad = True

            torch.manual_seed(self.seeds["FFN"])
            g_Y_1 = self.G(Y_1)

            torch.manual_seed(self.seeds["droppath"])
            g_Y_1 = drop_path(
                g_Y_1, drop_prob=self.drop_path_rate, training=self.training
            )

            g_Y_1.backward(dY_2, retain_graph=True)

        # activation recomputation is by design and not part of
        # the computation graph in forward pass.
        with torch.no_grad():
            X_2 = Y_2 - g_Y_1
            del g_Y_1

            dY_1 = dY_1 + Y_1.grad
            Y_1.grad = None

        # record F activations and calc gradients on F
        with torch.enable_grad():
            X_2.requires_grad = True

            torch.manual_seed(self.seeds["attn"])
            f_X_2 = self.F(X_2)

            torch.manual_seed(self.seeds["droppath"])
            f_X_2 = drop_path(
                f_X_2, drop_prob=self.drop_path_rate, training=self.training
            )

            f_X_2.backward(dY_1, retain_graph=True)

        # propagate reverse computed acitvations at the start of
        # the previou block for backprop.s
        with torch.no_grad():
            X_1 = Y_1 - f_X_2

            del f_X_2, Y_1
            dY_2 = dY_2 + X_2.grad

            X_2.grad = None
            X_2 = X_2.detach()

        return X_1, X_2, dY_1, dY_2


class MLPSubblock(nn.Module):
    """
    This creates the function G such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
            self,
            dim,
            mlp_ratio,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm = norm_layer(dim, eps=1e-6, elementwise_affine=True)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
        )

    def forward(self, x):
        return self.mlp(self.norm(x))


class AttentionSubBlock(nn.Module):
    """
    This creates the function F such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
            self,
            dim,
            input_size,
            num_heads,
            cfg,
            dim_out=None,
            kernel_q=(1, 1, 1),
            kernel_kv=(1, 1, 1),
            stride_q=(1, 1, 1),
            stride_kv=(1, 1, 1),
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm = norm_layer(dim, eps=1e-6, elementwise_affine=True)

        # This will be set externally during init
        self.thw = None

        # the actual attention details are the same as Multiscale
        # attention for MViTv2 (with channel up=projection inside block)
        # can also implement no upprojection attention for vanilla ViT
        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            input_size=input_size,
            num_heads=num_heads,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
            drop_rate=cfg.MVIT.DROPOUT_RATE,
            qkv_bias=cfg.MVIT.QKV_BIAS,
            has_cls_embed=cfg.MVIT.CLS_EMBED_ON,
            mode=cfg.MVIT.MODE,
            pool_first=cfg.MVIT.POOL_FIRST,
            rel_pos_spatial=cfg.MVIT.REL_POS_SPATIAL,
            rel_pos_temporal=cfg.MVIT.REL_POS_TEMPORAL,
            rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT,
            residual_pooling=cfg.MVIT.RESIDUAL_POOLING,
            separate_qkv=cfg.MVIT.SEPARATE_QKV,
        )

    def forward(self, x):
        out, _ = self.attn(self.norm(x), self.thw)
        return out


def cal_rel_pos_spatial(
        attn, q, k, has_cls_embed, q_shape, k_shape, rel_pos_h, rel_pos_w
):
    """
    Decomposed Spatial Relative Positional Embeddings.
    """
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dh = int(2 * max(q_h, k_h) - 1)
    dw = int(2 * max(q_w, k_w) - 1)

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
            torch.arange(q_h)[:, None] * q_h_ratio
            - torch.arange(k_h)[None, :] * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
            torch.arange(q_w)[:, None] * q_w_ratio
            - torch.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    # Intepolate rel pos if needed.
    rel_pos_h = get_rel_pos(rel_pos_h, dh)
    rel_pos_w = get_rel_pos(rel_pos_w, dw)
    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_t, q_h, q_w, dim)
    rel_h_q = torch.einsum(
        "bythwc,hkc->bythwk", r_q, Rh
    )  # [B, H, q_t, qh, qw, k_h]
    rel_w_q = torch.einsum(
        "bythwc,wkc->bythwk", r_q, Rw
    )  # [B, H, q_t, qh, qw, k_w]

    attn[:, :, sp_idx:, sp_idx:] = (
            attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
            + rel_h_q[:, :, :, :, :, None, :, None]
            + rel_w_q[:, :, :, :, :, None, None, :]
    ).view(B, -1, q_t * q_h * q_w, k_t * k_h * k_w)

    return attn


def cal_rel_pos_temporal(attn, q, has_cls_embed, q_shape, k_shape, rel_pos_t):
    """
    Temporal Relative Positional Embeddings.
    """
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dt = int(2 * max(q_t, k_t) - 1)
    # Intepolate rel pos if needed.
    rel_pos_t = get_rel_pos(rel_pos_t, dt)

    # Scale up rel pos if shapes for q and k are different.
    q_t_ratio = max(k_t / q_t, 1.0)
    k_t_ratio = max(q_t / k_t, 1.0)
    dist_t = (
            torch.arange(q_t)[:, None] * q_t_ratio
            - torch.arange(k_t)[None, :] * k_t_ratio
    )
    dist_t += (k_t - 1) * k_t_ratio
    Rt = rel_pos_t[dist_t.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_t, q_h, q_w, dim)
    # [B, H, q_t, q_h, q_w, dim] -> [q_t, B, H, q_h, q_w, dim] -> [q_t, B*H*q_h*q_w, dim]
    r_q = r_q.permute(2, 0, 1, 3, 4, 5).reshape(
        q_t, B * n_head * q_h * q_w, dim
    )

    # [q_t, B*H*q_h*q_w, dim] * [q_t, dim, k_t] = [q_t, B*H*q_h*q_w, k_t] -> [B*H*q_h*q_w, q_t, k_t]
    rel = torch.matmul(r_q, Rt.transpose(1, 2)).transpose(0, 1)
    # [B*H*q_h*q_w, q_t, k_t] -> [B, H, q_t, q_h, q_w, k_t]
    rel = rel.view(B, n_head, q_h, q_w, q_t, k_t).permute(0, 1, 4, 2, 3, 5)

    attn[:, :, sp_idx:, sp_idx:] = (
            attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
            + rel[:, :, :, :, :, :, None, None]
    ).view(B, -1, q_t * q_h * q_w, k_t * k_h * k_w)

    return attn


class Interpolate(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward(self, x):
        W, C = x.shape
        x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=2 * self.input_size[0] - 1, mode='nearest')
        x = x.permute(0, 2, 1)
        x = x.squeeze(0)

        return x


class MultiScaleAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            input_size,
            num_heads=8,
            qkv_bias=False,
            drop_rate=0.0,
            kernel_q=(1, 1, 1),
            kernel_kv=(1, 1, 1),
            stride_q=(1, 1, 1),
            stride_kv=(1, 1, 1),
            norm_layer=nn.LayerNorm,
            has_cls_embed=True,
            # Options include `conv`, `avg`, and `max`.
            mode="conv",
            # If True, perform pool before projection.
            pool_first=False,
            rel_pos_spatial=False,
            rel_pos_temporal=False,
            rel_pos_zero_init=False,
            residual_pooling=False,
            separate_qkv=False,
    ):
        super().__init__()
        self.pool_first = pool_first
        self.separate_qkv = separate_qkv
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        self.dim_out = dim_out
        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        self.mode = mode
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        if pool_first or separate_qkv:
            self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim_out, dim_out)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()

        if mode in ("avg", "max"):
            pool_op = nn.MaxPool3d if mode == "max" else nn.AvgPool3d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "conv" or mode == "conv_unshared":
            if pool_first:
                dim_conv = dim // num_heads if mode == "conv" else dim
            else:
                dim_conv = dim_out // num_heads if mode == "conv" else dim_out
            self.pool_q = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            self.norm_q = norm_layer(dim_conv) if len(kernel_q) > 0 else None
            self.pool_k = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_k = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
            self.pool_v = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_v = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        self.rel_pos_spatial = rel_pos_spatial
        self.rel_pos_temporal = rel_pos_temporal
        if self.rel_pos_spatial:
            assert input_size[1] == input_size[2]
            size = input_size[1]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1

            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)
        if self.rel_pos_temporal:
            self.rel_pos_t = nn.Parameter(
                torch.zeros(2 * 8 - 1, head_dim)
            )
            # self.upsample_t = Interpolate(input_size) if input_size[0] != 8 else nn.Identity()

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_t, std=0.02)

        self.residual_pooling = residual_pooling

    def forward(self, x, thw_shape):
        B, N, _ = x.shape

        if self.pool_first:
            if self.mode == "conv_unshared":
                fold_dim = 1
            else:
                fold_dim = self.num_heads
            x = x.reshape(B, N, fold_dim, -1).permute(0, 2, 1, 3)
            q = k = v = x
        else:
            assert self.mode != "conv_unshared"
            if not self.separate_qkv:
                qkv = (
                    self.qkv(x)
                    .reshape(B, N, 3, self.num_heads, -1)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, v = qkv[0], qkv[1], qkv[2]
            else:
                q = k = v = x
                q = (
                    self.q(q)
                    .reshape(B, N, self.num_heads, -1)
                    .permute(0, 2, 1, 3)
                )
                k = (
                    self.k(k)
                    .reshape(B, N, self.num_heads, -1)
                    .permute(0, 2, 1, 3)
                )
                v = (
                    self.v(v)
                    .reshape(B, N, self.num_heads, -1)
                    .permute(0, 2, 1, 3)
                )

        q, q_shape = attention_pool(
            q,
            self.pool_q,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_q", None),
        )
        k, k_shape = attention_pool(
            k,
            self.pool_k,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_k", None),
        )
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_v", None),
        )

        if self.pool_first:
            q_N = (
                numpy.prod(q_shape) + 1
                if self.has_cls_embed
                else numpy.prod(q_shape)
            )
            k_N = (
                numpy.prod(k_shape) + 1
                if self.has_cls_embed
                else numpy.prod(k_shape)
            )
            v_N = (
                numpy.prod(v_shape) + 1
                if self.has_cls_embed
                else numpy.prod(v_shape)
            )

            q = q.permute(0, 2, 1, 3).reshape(B, q_N, -1)
            q = (
                self.q(q)
                .reshape(B, q_N, self.num_heads, -1)
                .permute(0, 2, 1, 3)
            )

            v = v.permute(0, 2, 1, 3).reshape(B, v_N, -1)
            v = (
                self.v(v)
                .reshape(B, v_N, self.num_heads, -1)
                .permute(0, 2, 1, 3)
            )

            k = k.permute(0, 2, 1, 3).reshape(B, k_N, -1)
            k = (
                self.k(k)
                .reshape(B, k_N, self.num_heads, -1)
                .permute(0, 2, 1, 3)
            )

        N = q.shape[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rel_pos_spatial:
            attn = cal_rel_pos_spatial(
                attn,
                q,
                k,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_h,
                self.rel_pos_w,
            )

        if self.rel_pos_temporal:
            # W, C = self.rel_pos_t.shape
            # rel_pos_t = self.rel_pos_t.detach().clone().reshape(1,W,C)
            # rel_pos_t = rel_pos_t.permute(0, 2, 1)
            # rel_pos_t = F.interpolate(rel_pos_t, size=2 * 16 - 1, mode='nearest')
            # rel_pos_t = rel_pos_t.permute(0, 2, 1)
            # rel_pos_t = rel_pos_t[0]
            #
            # self.rel_pos_t = torch.nn.Parameter(rel_pos_t)

            # self.upsample_t(self.rel_pos_t)
            attn = cal_rel_pos_temporal(
                attn,
                q,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_t,
            )
        attn = attn.softmax(dim=-1)

        x = attn @ v

        if self.residual_pooling:
            if self.has_cls_embed:
                x[:, :, 1:, :] += q[:, :, 1:, :]
            else:
                x = x + q

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)

        if self.drop_rate > 0.0:
            x = self.proj_drop(x)
        return x, q_shape


class MultiScaleBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            num_heads,
            input_size,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.0,
            drop_path=0.0,
            layer_scale_init_value=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            up_rate=None,
            kernel_q=(1, 1, 1),
            kernel_kv=(1, 1, 1),
            stride_q=(1, 1, 1),
            stride_kv=(1, 1, 1),
            mode="conv",
            has_cls_embed=True,
            pool_first=False,
            rel_pos_spatial=False,
            rel_pos_temporal=False,
            rel_pos_zero_init=False,
            residual_pooling=False,
            dim_mul_in_att=False,
            separate_qkv=False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.dim_mul_in_att = dim_mul_in_att
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        att_dim = dim_out if dim_mul_in_att else dim
        self.attn = MultiScaleAttention(
            dim,
            att_dim,
            num_heads=num_heads,
            input_size=input_size,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
            has_cls_embed=has_cls_embed,
            mode=mode,
            pool_first=pool_first,
            rel_pos_spatial=rel_pos_spatial,
            rel_pos_temporal=rel_pos_temporal,
            rel_pos_zero_init=rel_pos_zero_init,
            residual_pooling=residual_pooling,
            separate_qkv=separate_qkv,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(att_dim)
        mlp_hidden_dim = int(att_dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=att_dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop_rate=drop_rate,
        )
        if layer_scale_init_value > 0:
            self.gamma_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim_out)),
                requires_grad=True,
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        self.pool_skip = (
            nn.MaxPool3d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )
            if len(stride_skip) > 0 and numpy.prod(stride_skip) > 1
            else None
        )

    def forward(self, x, thw_shape=None):
        x_norm = self.norm1(x)
        x_block, thw_shape_new = self.attn(x_norm, thw_shape)
        if self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        x_res, _ = attention_pool(
            x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed
        )
        if self.gamma_1 is not None:
            x = x_res + self.drop_path(self.gamma_1 * x_block)
        else:
            x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if not self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        if self.gamma_2 is not None:
            x = x + self.drop_path(self.gamma_2 * x_mlp)
        else:
            x = x + self.drop_path(x_mlp)
        if thw_shape:
            return x, thw_shape_new
        else:
            return x


def get_norm(cfg):
    """
    Args:
        cfg (CfgNode): model building configs, details are in the comments of
            the config file.
    Returns:
        nn.Module: the normalization layer.
    """
    if cfg.BN.NORM_TYPE in {"batchnorm", "sync_batchnorm_apex"}:
        return nn.BatchNorm3d
    else:
        raise NotImplementedError(
            "Norm type {} is not supported".format(cfg.BN.NORM_TYPE)
        )


# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {18: (2, 2, 2, 2), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "slow_c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow_i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}

_POOL1 = {
    "2d": [[1, 1, 1]],
    "c2d": [[2, 1, 1]],
    "slow_c2d": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "slow_i3d": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "x3d": [[1, 1, 1]],
}


def round_width(width, multiplier, min_width=1, divisor=1):
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor

    width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_3d_sincos_pos_embed(embed_dim, grid_size, t_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    # spatial
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(
        embed_dim_spatial, grid
    )

    # temporal
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(
        embed_dim_temporal, grid_t
    )

    # concate: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(
        pos_embed_temporal, grid_size ** 2, axis=1
    )  # [T, H*W, D // 4]
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(
        pos_embed_spatial, t_size, axis=0
    )  # [T, H*W, D // 4 * 3]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    pos_embed = pos_embed.reshape([-1, embed_dim])  # [T*H*W, D]

    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def calc_mvit_feature_geometry(cfg):
    feat_size = [
        [
            cfg.DATA.NUM_FRAMES // cfg.MVIT.PATCH_STRIDE[0]
            if len(cfg.MVIT.PATCH_STRIDE) > 2
            else 1,
            cfg.DATA.TRAIN_CROP_SIZE // cfg.MVIT.PATCH_STRIDE[-2],
            cfg.DATA.TRAIN_CROP_SIZE // cfg.MVIT.PATCH_STRIDE[-1],
        ]
        for i in range(cfg.MVIT.DEPTH)
    ]
    feat_stride = [
        [
            cfg.MVIT.PATCH_STRIDE[0] if len(cfg.MVIT.PATCH_STRIDE) > 2 else 1,
            cfg.MVIT.PATCH_STRIDE[-2],
            cfg.MVIT.PATCH_STRIDE[-1],
        ]
        for i in range(cfg.MVIT.DEPTH)
    ]
    for _, x in enumerate(cfg.MVIT.POOL_Q_STRIDE):
        for i in range(cfg.MVIT.DEPTH):
            if i >= x[0]:
                for j in range(len(feat_size[i])):
                    feat_size[i][j] = feat_size[i][j] // x[j + 1]
                    feat_stride[i][j] = feat_stride[i][j] * x[j + 1]
    return feat_size, feat_stride


class MViT(nn.Module):
    """
    Model builder for MViTv1 and MViTv2.

    "MViTv2: Improved Multiscale Vision Transformers for Classification and Detection"
    Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2112.01526
    "Multiscale Vision Transformers"
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, path_to_configs):
        super().__init__()
        print(path_to_configs)
        cfg = load_config(None, path_to_configs[0])
        cfg = assert_and_infer_cfg(cfg)
        # Get parameters.
        # assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        pool_first = cfg.MVIT.POOL_FIRST
        # Prepare input.
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        self.use_2d_patch = cfg.MVIT.PATCH_2D
        self.enable_detection = cfg.DETECTION.ENABLE
        self.enable_rev = cfg.MVIT.REV.ENABLE
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if self.use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        self.T = cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        self.H = cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
        self.W = cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]
        # Prepare output.
        num_classes = cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        layer_scale_init_value = cfg.MVIT.LAYER_SCALE_INIT_VALUE
        head_init_scale = cfg.MVIT.HEAD_INIT_SCALE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.use_mean_pooling = cfg.MVIT.USE_MEAN_POOLING
        # Params for positional embedding
        self.use_abs_pos = cfg.MVIT.USE_ABS_POS
        self.use_fixed_sincos_pos = cfg.MVIT.USE_FIXED_SINCOS_POS
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        self.rel_pos_spatial = cfg.MVIT.REL_POS_SPATIAL
        self.rel_pos_temporal = cfg.MVIT.REL_POS_TEMPORAL
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=self.use_2d_patch,
        )

        # if cfg.MODEL.ACT_CHECKPOINT:
        #     self.patch_embed = checkpoint_wrapper(self.patch_embed)
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(
                        1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                    )
                )
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim)
                )
                if self.cls_embed_on:
                    self.pos_embed_class = nn.Parameter(
                        torch.zeros(1, 1, embed_dim)
                    )
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(
                        1,
                        pos_embed_dim,
                        embed_dim,
                    ),
                    requires_grad=not self.use_fixed_sincos_pos,
                )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][
                                                     1:
                                                     ]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[
                                                           i
                                                       ][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]
                ]

        self.pool_q = pool_q
        self.pool_kv = pool_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv

        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        input_size = self.patch_dims

        if self.enable_rev:

            # rev does not allow cls token
            assert not self.cls_embed_on

            self.rev_backbone = ReversibleMViT(cfg, self)

            embed_dim = round_width(
                embed_dim, dim_mul.prod(), divisor=num_heads
            )

            self.fuse = TwoStreamFusion(
                cfg.MVIT.REV.RESPATH_FUSE, dim=2 * embed_dim
            )

            if "concat" in self.cfg.MVIT.REV.RESPATH_FUSE:
                self.norm = norm_layer(2 * embed_dim)
            else:
                self.norm = norm_layer(embed_dim)

        else:

            self.blocks = nn.ModuleList()

            for i in range(depth):
                num_heads = round_width(num_heads, head_mul[i])
                if cfg.MVIT.DIM_MUL_IN_ATT:
                    dim_out = round_width(
                        embed_dim,
                        dim_mul[i],
                        divisor=round_width(num_heads, head_mul[i]),
                    )
                else:
                    dim_out = round_width(
                        embed_dim,
                        dim_mul[i + 1],
                        divisor=round_width(num_heads, head_mul[i + 1]),
                    )
                attention_block = MultiScaleBlock(
                    dim=embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    input_size=input_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=self.drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    kernel_q=pool_q[i] if len(pool_q) > i else [],
                    kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                    stride_q=stride_q[i] if len(stride_q) > i else [],
                    stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                    mode=mode,
                    has_cls_embed=self.cls_embed_on,
                    pool_first=pool_first,
                    rel_pos_spatial=self.rel_pos_spatial,
                    rel_pos_temporal=self.rel_pos_temporal,
                    rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT,
                    residual_pooling=cfg.MVIT.RESIDUAL_POOLING,
                    dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,
                    separate_qkv=cfg.MVIT.SEPARATE_QKV,
                )

                # if cfg.MODEL.ACT_CHECKPOINT:
                #     attention_block = checkpoint_wrapper(attention_block)
                self.blocks.append(attention_block)
                if len(stride_q[i]) > 0:
                    input_size = [
                        size // stride
                        for size, stride in zip(input_size, stride_q[i])
                    ]

                embed_dim = dim_out

            self.norm = norm_layer(embed_dim)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                trunc_normal_(self.pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)
                if self.use_fixed_sincos_pos:
                    pos_embed = get_3d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        self.H,
                        self.T,
                        cls_token=self.cls_embed_on,
                    )
                    self.pos_embed.data.copy_(
                        torch.from_numpy(pos_embed).float().unsqueeze(0)
                    )

        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        self.feat_size, self.feat_stride = calc_mvit_feature_geometry(cfg)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.02)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.use_abs_pos:
                if self.sep_pos_embed:
                    names.extend(
                        [
                            "pos_embed_spatial",
                            "pos_embed_temporal",
                            "pos_embed_class",
                        ]
                    )
                else:
                    names.append("pos_embed")
            if self.rel_pos_spatial:
                names.extend(["rel_pos_h", "rel_pos_w", "rel_pos_hw"])
            if self.rel_pos_temporal:
                names.extend(["rel_pos_t"])
            if self.cls_embed_on:
                names.append("cls_token")

        return names

    def _get_pos_embed(self, pos_embed, bcthw):

        if len(bcthw) == 4:
            t, h, w = 1, bcthw[-2], bcthw[-1]
        else:
            t, h, w = bcthw[-3], bcthw[-2], bcthw[-1]
        if self.cls_embed_on:
            cls_pos_embed = pos_embed[:, 0:1, :]
            pos_embed = pos_embed[:, 1:]
        txy_num = pos_embed.shape[1]
        p_t, p_h, p_w = self.patch_dims
        assert p_t * p_h * p_w == txy_num

        if (p_t, p_h, p_w) != (t, h, w):
            new_pos_embed = F.interpolate(
                pos_embed[:, :, :]
                .reshape(1, p_t, p_h, p_w, -1)
                .permute(0, 4, 1, 2, 3),
                size=(t, h, w),
                mode="trilinear",
            )
            pos_embed = new_pos_embed.reshape(1, -1, t * h * w).permute(0, 2, 1)

        if self.cls_embed_on:
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)

        return pos_embed

    def _forward_reversible(self, x):
        """
        Reversible specific code for forward computation.
        """
        # rev does not support cls token or detection
        assert not self.cls_embed_on
        assert not self.enable_detection

        x = self.rev_backbone(x)

        if self.use_mean_pooling:
            x = self.fuse(x)
            x = x.mean(1)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.fuse(x)
            x = x.mean(1)

        x = self.head(x)

        return x

    def forward(self, x, bboxes=None, return_attn=False):
        x = x[0]
        x, bcthw = self.patch_embed(x)
        bcthw = list(bcthw)
        if len(bcthw) == 4:  # Fix bcthw in case of 4D tensor
            bcthw.insert(2, torch.tensor(self.T))
        T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
        # assert len(bcthw) == 5 and (T, H, W) == (self.T, self.H, self.W), bcthw
        B, N, C = x.shape

        s = 1 if self.cls_embed_on else 0
        if self.use_fixed_sincos_pos:
            x += self.pos_embed[:, s:, :]  # s: on/off cls token

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            if self.use_fixed_sincos_pos:
                cls_tokens = cls_tokens + self.pos_embed[:, :s, :]
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x += self._get_pos_embed(pos_embed, bcthw)
            else:
                x += self._get_pos_embed(self.pos_embed, bcthw)

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]

        feas = []
        marks = {0, 2, 13, 15}
        settings = [56, 28, 14, 7]
        ct = 0

        if self.enable_rev:
            x = self._forward_reversible(x)
        else:
            for i, blk in enumerate(self.blocks):
                x, thw = blk(x, thw)
                # print(i,x.shape)
                if i in marks:
                    feas.append(rearrange(x, 'b (t h w) c -> b c t h w', t=T, h=settings[ct]))
                    ct += 1

        return feas

    def load_weight(self, path):
        weight = torch.load(path, map_location='cpu')['model_state']
        self.load_state_dict(weight, strict=False)
        print("MViTv2 Weight Loaded!")


