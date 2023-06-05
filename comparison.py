import torch

from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_argument_parser
from detectron2.checkpoint import DetectionCheckpointer

from vit import Int8Attention, Int8Block
from functools import partial

from torch_int.nn.linear import W8A8B8O8Linear
from torch import nn

# Load the model
# Get the first layer
# Load a input data
# Get the output
# Compare with quantized module

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    """Takes the original layers in (in fp16 format) and adjustes the ln and
    linear layers. Returns still an fp16 format.

    In evaluation, the results should probably be unchanged?
    """
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat([fc.weight.abs().max(
        dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    #has the shape out in_features

    scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)
              ).clamp(min=1e-5).to(device).to(dtype)

    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

def load(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    model = instantiate(cfg.model)
    model = model.to('cuda')
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    block = model.backbone.net.blocks[0]
    x = torch.load('/tmp/x.pt')
    return block, x

def store_act(module, x, y, act_dict, name):
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(y, tuple):
        y = y[0]
    act_dict[name] = (x, y)


def smooth_module(module, x):
    alpha = 0.5
    module.eval()
    act_dict = {}
    for n, m in module.named_modules():
        if isinstance(m, torch.nn.Linear):
            m.register_forward_hook(
                partial(store_act, act_dict=act_dict, name=n))
    for n, m in module.named_modules():
        if isinstance(m, torch.nn.Linear):
            m.register_forward_hook(
                partial(store_act, act_dict=act_dict, name=n))
    # x_scale = x.abs().max() / 127
    module(x)
    qkv_input = act_dict['attn.qkv'][0]
    qkv_input = qkv_input.view(-1, 1024).abs().detach()
    qkv_input = torch.max(qkv_input, dim=0)[0]
    attn_ln = module.norm1
    qkv = module.attn.qkv
    smooth_ln_fcs(attn_ln, qkv, qkv_input, alpha)
    ffn_ln = module.norm2
    fc1 = module.mlp.fc1
    fc1_input_scales = act_dict['mlp.fc1'][0]
    fc1_input_scales = fc1_input_scales.view(-1, 1024).abs().detach()
    fc1_input_scales= torch.max(fc1_input_scales, dim=0)[0]
    smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)


def generate_quantized_module(module, x):
    module.eval()
    act_dict = {}
    for n, m in module.named_modules():
        if isinstance(m, torch.nn.Linear):
            m.register_forward_hook(
                partial(store_act, act_dict=act_dict, name=n))
    x_scale = x.abs().max() / 127
    y = module(x)
    qkv = act_dict['attn.qkv'][1]
    out_features = qkv.shape[-1] // 3
    q_output_scale = qkv[:, :, :, :out_features].abs().max() / 127
    k_output_scale = qkv[:, :, :, out_features: 2 * out_features].abs().max() / 127
    v_output_scale = qkv[:, :, :, 2 * out_features: ].abs().max() / 127
    proj_input_scale = act_dict['attn.proj'][0].abs().max() / 127
    breakpoint()
    # fc1_input_scale = 1.0
    # fc2_input_scale = 1.0
    fc1_input_scale = act_dict['mlp.fc1'][0].abs().max() / 127
    fc2_input_scale = act_dict['mlp.fc2'][0].abs().max() / 127
    block = Int8Block.from_float(module,
            attn_input_scale=x_scale,
            q_output_scale=q_output_scale,
            k_output_scale=k_output_scale,
            v_output_scale=v_output_scale,
            proj_input_scale=proj_input_scale,
            fc1_input_scale=fc1_input_scale,
            fc2_input_scale=fc2_input_scale)
    # q_x = (x / x_scale).round().to(torch.int8).to('cuda')
    y_hat = block(x)
    breakpoint()
    diff = y - y_hat
    return diff

# def compare(attention_orig, attention_new, x):
    # y_orig = attention_orig(x)
    # y = attention_new(x)
    # return y_orig, y

def main():
    args = default_argument_parser().parse_args()
    orig_block, x = load(args)
    smooth_module(orig_block,x )
    generate_quantized_module(orig_block, x)
    # attn_input_scale = 0.0457
    # qkv_output_scale = 0.0256
    # proj_input_scale = 0.0162
    # fc1_input_scale = 0.0433
    # fc2_input_scale = 0.0248
    # breakpoint()
    # q_block = load_quantized_module(orig_block, attn_input_scale,
            # qkv_output_scale, proj_input_scale, fc1_input_scale,
            # fc2_input_scale)
    # compare(orig_block, q_block, x)

if __name__ == "__main__":
    main()
