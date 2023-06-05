r"""
Has a quantized VIT
"""
import torch
from torch import nn

from torch_int.nn.linear import (
        W8A8BFP32OFP32Linear,
        W8A8B8O8Linear,
        W8A8B8O8LinearReLU,
        W8A8B8O8LinearGELU,
        W8A8BFP32OFP32LinearGELU
        )

from torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T
from torch_int.nn.fused import LayerNormQ

from detectron2.modeling.backbone.vit import Attention, Block
from detectron2.modeling.backbone.utils import (
        window_partition,
        window_unpartition,
        add_decomposed_rel_pos)


class Int8Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim: int, num_heads: int, use_rel_pos: bool,
            input_size = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )

        self.attention_weight_scale = 1.0

        self.qk_bmm = BMM_S8T_S8N_F32T(1.0)
        self.pv_bmm = BMM_S8T_S8N_S8T(1.0)

        # self.qkv = W8A8B8O8Linear(embed_dim, 3 * embed_dim)
        self.q = W8A8B8O8Linear(embed_dim, embed_dim)
        self.k = W8A8B8O8Linear(embed_dim, embed_dim)
        self.v = W8A8B8O8Linear(embed_dim, embed_dim)
        self.proj = W8A8BFP32OFP32Linear(embed_dim, embed_dim)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # added to the (unnformalized) attention weight.s Keep in float 
            # to simipliciy, as the unnformalized weights are also in float
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.head_dim))

    @staticmethod
    @torch.no_grad()
    def from_float(module: Attention,
                   input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   proj_input_scale: float):
        int8_module = Int8Attention(module.qkv.in_features, module.num_heads,
                use_rel_pos=True,
                input_size=(14, 14))
        in_features = module.qkv.in_features
        out_features = module.qkv.out_features  // 3
        q_module = nn.Linear(in_features=in_features,
                out_features=out_features,device='cuda')
        k_module = nn.Linear(in_features=in_features,
                out_features=out_features,
                device='cuda')
        v_module = nn.Linear(in_features=in_features,
                out_features=out_features,
                device='cuda')
        q_module.weight = torch.nn.Parameter(module.qkv.weight[:out_features, :])
        q_module.bias = torch.nn.Parameter(module.qkv.bias[:out_features])
        k_module.weight = torch.nn.Parameter(module.qkv.weight[out_features:2*out_features, :])
        k_module.bias = torch.nn.Parameter(module.qkv.bias[out_features: 2*out_features])
        v_module.weight = torch.nn.Parameter(module.qkv.weight[2 * out_features:, :])
        v_module.bias = torch.nn.Parameter(module.qkv.bias[2 * out_features:])
        # Fuse the scaling into the q_proj output scale
        # The scale is the qkv / scale thing
        # qkv_output_scale = qkv_output_scale * module.scale
        # int8_module.qkv = W8A8B8O8Linear.from_float(
                # module.qkv, input_scale, q_output_scale)
        q_output_scale *= module.scale
        q_module.weight *= module.scale
        q_module.bias *= module.scale
        breakpoint()
        # module.proj.weight *= module.scale
        # module.proj.bias *= module.scale
        # qkv_output_scale = qkv_output_scale
        #Seperate three linear layers, in particular v has a different scale
        int8_module.q = W8A8B8O8Linear.from_float(
            q_module, input_scale, q_output_scale)
        int8_module.k = W8A8B8O8Linear.from_float(
            k_module, input_scale, k_output_scale)
        int8_module.v = W8A8B8O8Linear.from_float(
            v_module, input_scale, v_output_scale)
        int8_module.proj = W8A8BFP32OFP32Linear.from_float(
            module.proj, proj_input_scale)
        int8_module.qk_bmm = BMM_S8T_S8N_F32T.from_scale(
            q_output_scale, k_output_scale)
        #attn_input_scale: 0.0415
        #k_output_scale: 0.067
        #v_output_scale: 0.066
        #q_output_scale: 0.0103
        #fuse the module.scale (scale for q @ k) into the operator

        if int8_module.use_rel_pos:
            int8_module.rel_pos_h = module.rel_pos_h
            int8_module.rel_pos_w = module.rel_pos_w
            # not sure if the same scales make sense...

        # alpha = s_prob * s_v / s_out, where s_prob = 1 / 127
        int8_module.pv_bmm = BMM_S8T_S8N_S8T.from_scale(
            1.0 / 127, v_output_scale, proj_input_scale)
        return int8_module

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x): #25x14x14x1024 (for large)
        B, H, W, _ = x.shape #H, W: Number of patches
        q = self.q(x).reshape(B, H * W, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B * self.num_heads, H * W, -1)
        k = self.k(x).reshape(B, H * W, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B * self.num_heads, H * W, -1)

        attn = self.qk_bmm(q, k)
        #attn shape: 400x196x196

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, (q * 0.01),
                                          self.rel_pos_h, self.rel_pos_w,
                                          (H, W), (H, W))

        attn_probs = nn.functional.softmax(attn, dim=-1) #why not take max for scaling too?
        attn_probs.mul_(127).round_()
        attn_probs = attn_probs.to(torch.int8)

        #different layout because pv_bmm takes a col major matrix as second arg
        v = self.v(x).reshape(B, H * W, self.num_heads, -1)\
                .permute(0, 2, 3, 1)\
                .reshape(B * self.num_heads, 64, H * W)
        anew = torch.zeros((B * self.num_heads, 196, 224), dtype=torch.int8, device='cuda')
        bnew = torch.zeros((B * self.num_heads, 64, 224), dtype=torch.int8, device='cuda')
        bnew[:, :, :196] = v
        anew[:, :, :196] = attn_probs
        x = self.pv_bmm(anew, bnew)

        x = x.reshape(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x


class Int8Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        use_residual_block=False,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = LayerNormQ(dim)
        self.attn = Int8Attention(
            dim,
            num_heads=num_heads,
            use_rel_pos=True,
            input_size=(14, 14))
            # qkv_bias=qkv_bias)
            # use_rel_pos=use_rel_pos,
            # rel_pos_zero_init=rel_pos_zero_init,
            # input_size=input_size if window_size == 0 else (window_size, window_size),
        # )

        from timm.models.layers import DropPath, Mlp

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = LayerNormQ(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)
        # del self.mlp.act

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        assert not self.use_residual_block, "residual block not yet supported"
        # if use_residual_block:
            # # Use a residual block with bottleneck channel as dim // 2
            # self.residual = ResBottleneckBlock(
                # in_channels=dim,
                # out_channels=dim,
                # bottleneck_channels=dim // 2,
                # norm="LN",
                # act_layer=act_layer,
            # )

    @staticmethod
    def from_float(module: Block,
                   attn_input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   proj_input_scale: float,
                   fc1_input_scale: float,
                   fc2_input_scale: float):
        #this is equation 1 from the paper.
        int8_module = Int8Block(
            module.attn.qkv.in_features,
            module.attn.num_heads,
            window_size=module.window_size)
            # module.attn.qkv.out_features,
        int8_module.norm1 = LayerNormQ.from_float(module.norm1, attn_input_scale)
        int8_module.attn = Int8Attention.from_float(
            module.attn, attn_input_scale, q_output_scale, k_output_scale,
            v_output_scale, proj_input_scale)
        breakpoint()
        int8_module.norm2 = LayerNormQ.from_float(
            module.norm2, fc1_input_scale)
        int8_module.mlp.fc1 = W8A8BFP32OFP32Linear.from_float(
            module.mlp.fc1, fc1_input_scale)
        # int8_module.mlp.fc1 = W8A8B8O8LinearGELU.from_float(
            # module.mlp.fc1, fc1_input_scale, fc2_input_scale)
        int8_module.mlp.fc2 = module.mlp.fc2
        # int8_module.mlp.fc2 = W8A8BFP32OFP32Linear.from_float(
            # module.mlp.fc2, fc2_input_scale)
        return int8_module

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x) #2% error
        breakpoint()
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x_tmp = x
        x = self.norm2(x)
        x = self.mlp.fc1(x)
        x = self.mlp.act(x)
        x = self.mlp.drop1(x)
        x = self.mlp.fc2(x)
        x = self.mlp.drop2(x)
        x =  x_tmp + self.drop_path(x)

        # x = shortcut + self.drop_path(x)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x
