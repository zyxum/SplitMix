import torch

from nets.slimmable_ops import SlimmableConv2d, SlimmableLinear, SlimmableBatchNorm2d

import sys, pickle
from collections import OrderedDict
from torch.nn import Module
import torch.nn as nn
from thop import profile


def get_layer_names(state_dict: OrderedDict):
    keys = state_dict.keys()
    layers = set()
    first = None
    last = None
    # print(keys)
    for key in keys:
        if len(key.split('.')) <= 1:
            continue
        if key.split('.')[-1] not in ['weight', 'bias']:
            continue
        layer_name = ".".join(key.split('.')[:-1])
        if first is None:
            first = layer_name
        last = layer_name
        layers.add(layer_name)
    return list(layers), first, last

def get_layer(model: Module, layer_name: str):
    # print(layer_name)
    attributes = layer_name.split('.')
    tmp = model
    # print(layer_name)
    for attribute in attributes:
        tmp = getattr(tmp, attribute)
    return tmp

def set_layer(model: Module, layer_name: str, layer: Module):
    attributes = layer_name.split('.')
    tmp = model
    for idx, attribute in enumerate(attributes):
        if idx == len(attributes) - 1:
            break
        tmp = getattr(tmp, attribute)
    setattr(tmp, attributes[-1], layer)

def replace(model_path) -> nn.Module:
    # model_path = sys.argv[1]
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    # print("brefore replacing:")
    # print(model)
    # print("###"*100)
    macs, _ = profile(model, inputs=(torch.randn(10, 1, 28, 28),), verbose=False)
    print(macs)
    layers, first_layer, last_layer = get_layer_names(model.state_dict())
    for layer_name in layers:
        layer = get_layer(model, layer_name)
        if isinstance(layer, nn.Conv2d):
            new_layer = SlimmableConv2d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=True if layer.bias is not None else False,
                non_slimmable_in=layer_name==first_layer,
                non_slimmable_out=layer_name==last_layer
            )
        elif isinstance(layer, nn.Linear):
            new_layer = SlimmableLinear(
                in_features=layer.in_features,
                out_features=layer.out_features,
                bias=True if layer.bias is not None else False,
                non_slimmable_in=layer_name==first_layer,
                non_slimmable_out=layer_name==last_layer
            )
        elif isinstance(layer, nn.BatchNorm2d):
            new_layer = SlimmableBatchNorm2d(
                num_features=layer.num_features,
                eps=layer.eps,
                momentum=layer.momentum,
                affine=layer.affine,
                track_running_stats=False
            )
        else:
            raise RuntimeError(f"unsupported layer type in replacing {type(layer)}")

        set_layer(model, layer_name, new_layer)
    return model

def main():
    model = replace('model_19.pth.tar')
    macs, _ = profile(model, inputs=(torch.randn(10, 3, 28, 28),), verbose=False)
    print(macs)
    print(model)

if __name__ == "__main__":
    main()