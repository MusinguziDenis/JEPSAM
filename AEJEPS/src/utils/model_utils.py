from addict import Dict

import logging

import torch.nn as nn
from typing import Type
import Levenshtein

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



def indices_to_chars(indices, vocab):
    tokens = []
    
    for i in indices: # This loops through all the indices
        if int(i) == vocab.TOKENS_MAPPING["[SOS]"]: # If SOS is encountered, dont add it to the final list
            continue
        elif int(i) == vocab.TOKENS_MAPPING["[EOS]"] or int(i) == vocab.TOKENS_MAPPING["[PAD]"]: # If EOS is encountered, stop the decoding process
            break
        else:
            tokens.append(vocab.REVERSE_TOKENS_MAPPING[int(i)])
    return tokens    
        

def calc_edit_distance(predictions, y, ly, vocab, print_example= True):

    dist                = 0
    batch_size, seq_len = predictions.shape

    for batch_idx in range(batch_size): 

        y_sliced    = indices_to_chars(y[batch_idx,0:ly[batch_idx]], vocab)
        pred_sliced = indices_to_chars(predictions[batch_idx], vocab)
        # print()
        # print("Ground Truth : ", y_sliced)
        # print("Prediction   : ", pred_sliced)

        # Strings - When you are using characters from the AudioDataset
        y_string    = ''.join(y_sliced)
        pred_string = ''.join(pred_sliced)
        
        dist        += Levenshtein.distance(pred_string, y_string)
        # Comment the above and uncomment below for toy dataset, as the toy dataset has a list of phonemes to compare
        # dist      += Levenshtein.distance(y_sliced, pred_sliced)

    if print_example: 
        # Print y_sliced and pred_sliced if you are using the toy dataset
        print()
        print("Ground Truth : ", y_sliced)
        print("Prediction   : ", pred_sliced)
        
    dist/=batch_size
    return dist