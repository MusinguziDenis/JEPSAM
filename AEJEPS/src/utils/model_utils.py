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

class TFScheduler:
  """
    Author: Vish Srinivasam (11785 - IDL TA@CMU, Fall 2022 - Spring 2023) 
  """
  def __init__(
      self, 
      mode, 
      factor, 
      init_val=1, 
      threshold=30, 
      cooldown=7, 
      restart_threshold=5, 
      patience=5, 
      cycle_duration=25
    ):
    self.init_val = init_val
    self.threshold = threshold
    self.cooldown = 0
    self.max_cooldown = cooldown
    self.mode = mode
    self.val = init_val
    self.factor = factor
    self.cycle = 1
    self.init_restart_threshold = restart_threshold
    self.restart_threshold = restart_threshold
    self.patience = patience
    self.patience_count = 0
    self.prev_dist = None
    self.curr_dist = None
    self.cycle_duration = cycle_duration
    self.cycle_cooldown = 0
    self.limit = 0.5
    self.counter = 0
  
  def config(self):
    tf_config = {}
    tf_config['init_val'] = self.init_val
    tf_config['threshold'] = self.threshold
    tf_config['factor'] = self.factor
    tf_config['mode'] = self.mode
    
    if self.mode == 'rop':  
      tf_config['patience'] = self.patience
    
    elif self.mode == 'step':
      tf_config['cooldown'] = self.max_cooldown

    elif self.mode == 'exp_restart':
      tf_config['limit'] = self.limit

    elif self.mode == 'step_restart':  
      tf_config['restart_threshold'] = self.init_restart_threshold
      tf_config['cooldown'] = self.max_cooldown

    else:
      tf_config['cycle_duration'] = self.cycle_duration
      tf_config['cycle_cooldown'] = self.cycle_cooldown
      
    return tf_config

  def step(self, val_ldist, epoch=None):
    
    if self.mode == 'rop':
      if self.curr_dist is None:
        self.curr_dist = val_ldist
      else:
        self.prev_dist = self.curr_dist
        self.curr_dist = val_ldist
      if val_ldist < self.threshold:
        if self.prev_dist is not None:
          if self.curr_dist > self.prev_dist:
            self.patience_count += 1
        if self.patience_count > self.patience:
          self.patience_count = 0
          self.val *= self.factor
        if self.val < 0.7:
          self.val = self.init_val
    
    elif self.mode == 'custom':
      lim_dist = 25
      
      if self.val == 1 and val_ldist < 500:
        self.val = 0.9
      if val_ldist <= lim_dist and self.counter == 4 and self.val >= 0.25:
        self.val -= 0.05
        self.counter = 0
      elif val_ldist <= lim_dist and self.counter != 4 and self.val >= 0.25:
        self.counter += 1
      
    elif self.mode == 'exp_restart':
      if val_ldist < self.threshold:
        self.val = self.val*self.factor
        if self.val < self.limit:
          self.val = self.init_val
    
    elif self.mode == 'exp':
      if val_ldist < self.threshold and self.val*self.factor > self.limit:
        self.val = self.val*self.factor
        # print("Exp Step")

    elif self.mode == 'step':
      if val_ldist < self.threshold and self.cooldown==0:
        self.val = self.val*self.factor
        self.cooldown = self.max_cooldown
      self.cooldown -= 1

    elif self.mode == 'step_restart':
      # #Step Restart TF scheduling
      if self.restart_threshold == 0:
        self.val = self.init_val
        self.restart_threshold = self.init_restart_threshold
      if val_ldist < self.threshold and self.cooldown==0:
        self.val = self.val*self.factor
        self.restart_threshold -=1
        self.cooldown = self.max_cooldown
      self.cooldown -= 1

      # #Cyclic Teacher Force Scheduling from 1 to 
    elif self.mode == 'cycle':
      if val_ldist < self.threshold and self.cycle_cooldown < self.cycle_duration and self.cycle == 1:
        self.val = self.val*self.factor
        self.cycle_cooldown +=1
        if self.cycle_cooldown == self.cycle_duration:
          self.cycle = 0
          self.cycle_cooldown = self.cycle_duration*2
      elif val_ldist < self.threshold and self.cycle_cooldown >= self.cycle_duration and self.cycle == 0:
        self.val = self.val / self.factor
        self.cycle_cooldown -= 1
        if self.cycle_cooldown == self.cycle_duration:
          self.cycle = 1
    
    elif self.mode == 'cycle_decay':
      if val_ldist < self.threshold and self.cycle_cooldown < self.cycle_duration and self.cycle == 1:
        self.val = self.val*self.factor
        self.cycle_cooldown +=1
        if self.cycle_cooldown == self.cycle_duration:
          self.cycle = 0
          self.cycle_duration /= 2
          self.cycle_cooldown = self.cycle_duration*2
      elif val_ldist < self.threshold and self.cycle_cooldown >= self.cycle_duration and self.cycle == 0:
        self.val = self.val / self.factor
        self.cycle_cooldown -= 1
        if self.cycle_cooldown == self.cycle_duration:
          self.cycle = 1
          self.cycle_cooldown = 0
    
    elif self.mode == 'step_cycle':
      if val_ldist < self.threshold and self.cycle_cooldown < self.cycle_duration and self.cycle == 1 and self.cooldown == 0:
        self.val = self.val*self.factor
        self.cooldown = self.max_cooldown
        self.cycle_cooldown +=1
        if self.cycle_cooldown == self.cycle_duration:
          self.cycle = 0
          self.cycle_cooldown = self.cycle_duration*2
      elif val_ldist < self.threshold and self.cycle_cooldown >= self.cycle_duration and self.cycle == 0 and self.cooldown == 0:
        self.val = self.val / self.factor
        self.cooldown = self.max_cooldown
        self.cycle_cooldown -= 1
        if self.cycle_cooldown == self.cycle_duration:
          self.cycle = 1
      self.cooldown -= 1

    return self.val

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