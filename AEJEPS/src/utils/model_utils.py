from addict import Dict

import logging

import torch.nn as nn
from typing import Type

import torchvision.models as torchvision_models

import parser


logging.basicConfig(level="INFO")

def freeze_module(module: Type[nn.Module]):
    """
    Freezes the parameters of a module so gradient will not be computed for them.

    Parameters
    ----------
    module : torch.nn.Module
        Any subclass of torch.nn.Module

    Returns
    -------

    """
    for param in module.parameters():
        param.requires_grad = False
        

def get_cnn_backbone(
    cfg:Dict=None, 
    backbone_name:str="resnet50", 
    freeze:bool=True
):
    
    if cfg == None:
        cfg = parser.load_config()

    try:
        backbone = getattr(torchvision_models, backbone_name)(weights=cfg.MODEL.CNN_BACKBONES[backbone_name])
        logging.info(f"Successfully loaded CNN backbone: {backbone_name}")
    except Exception as e:
        logging.error(e)
        raise(e)


    # freeze backbone if specified
    if freeze:
        # for param in backbone.parameters():
        #     param.requires_grad = False
        freeze_module(backbone)

    # resnet-based models
    if "resnet" in backbone_name.lower():
        backbone.fc = nn.Linear(in_features=backbone.fc.in_features, out_features=cfg.AEJEPS.CNN_FC_DIM)
    
    return backbone


if __name__ == '__main__':
    import torch.nn as nn

    m = get_cnn_backbone(backbone_name="resnet18")
    freeze_module(m)

    print(m)

    all_params = set([p.requires_grad for p in m.parameters()])

    if len(all_params) != 1:
        print(f"Test failed: expected 'all_params' to contain only False values but contains {all_params}")
    else:
        print("Test passed!")