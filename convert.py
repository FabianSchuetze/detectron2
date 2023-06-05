# Load the model
from collections import defaultdict
from functools import partial
from tqdm import tqdm

import torch
import numpy as np

from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_argument_parser
from detectron2.checkpoint import DetectionCheckpointer

@torch.no_grad()
def get_static_decoder_layer_scales(model, dataset, vit_blocks: int):
    ## What is this needed for?
    model.eval()
    device = next(model.parameters()).device
    model.to(device)

    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = x.detach().abs().max().item()
        else:
            act_dict[name]["input"] = max(
                act_dict[name]["input"], x.detach().abs().max().item())
        if isinstance(y, tuple):
            y = y[0]
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = y.detach().abs().max().item()
        else:
            act_dict[name]["output"] = max(
                act_dict[name]["output"], y.detach().abs().max().item())

    hooks = []
    for name, m in model.named_modules():
        # Registering a hook. What is this doing?
        if isinstance(m, torch.nn.Linear) and 'backbone' in name:
            print(name)
            hooks.append(m.register_forward_hook(
                partial(stat_io_hook, name=name)))

    print("Collecting activation scales...")
    # pbar = tqdm(range(len(dataset)))
    pbar = tqdm(enumerate(dataset))
    for idx, inputs in pbar:
    # for i in pbar:
        # input_ids = tokenizer(dataset[i]["text"], return_tensors="pt",
                              # max_length=seq_len, truncation=True).input_ids.to(device)
        model(inputs)
        mean_scale = np.mean([v["input"] for v in act_dict.values()])
        pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
        if idx > 256:
            break
    for hook in hooks:
        hook.remove()

    decoder_layer_scales = []
    for idx in range(vit_blocks):
        scale_dict = {}
        scale_dict["attn_input_scale"] = act_dict[
            f"backbone.net.blocks.{idx}.attn.qkv"]['input'] / 127
        scale_dict["qkv_output_scale"] = act_dict[
            f"backbone.net.blocks.{idx}.attn.qkv"]['output'] / 127
        scale_dict["proj_input_scale"] = act_dict[
            f"backbone.net.blocks.{idx}.attn.proj"]['input'] / 127
        scale_dict["fc1_input_scale"] = act_dict[
            f"backbone.net.blocks.{idx}.mlp.fc1"]['input'] / 127
        scale_dict["fc2_input_scale"] = act_dict[
            f"backbone.net.blocks.{idx}.mlp.fc2"]["input"] / 127
        decoder_layer_scales.append(scale_dict)
        breakpoint()

    return decoder_layer_scales, act_dict


def load(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    model = instantiate(cfg.model)
    model = model.to('cuda')
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    dataset = instantiate(cfg.dataloader.test)
    return model, dataset

def main():
    args = default_argument_parser().parse_args()
    model, dataset = load(args)
    VIT_BLOCKS = 12
    decoder_layer_scales, act_dict = get_static_decoder_layer_scales(model,
            dataset, VIT_BLOCKS)


if __name__ == "__main__":
    main()
