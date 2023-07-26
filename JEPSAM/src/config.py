import os
from pprint import pprint

import sys
import torch as th

# # add modules to pythonPath
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

import vocabulary as vocab

class Config:

    # I/O params
    MODEL_ZOO                    = os.path.join(PROJECT_DIR, "models")
    LOGS_DIR                     = os.path.join(PROJECT_DIR, "logs")
    EXP_NAME                     = "unicorn-v1"
    IMAGE_SIZE                   = 128  # Image size
    DATASET = {
        "PATH"                      : os.path.join(PROJECT_DIR, "..", "dataset/"),         
        "TARGET_VOCABULARY_SIZE"    : len(vocab.MOTOR_COMMANDS), # number of unique tokens in the vocabulary (cmds)
        "SOS"                       : 48,   # [SOS] token index in vocabulary
        "EOS"                       : 46,   # [EOS] token index in vocabulary
        "PAD"                       : 47,   # [PAD] token index in vocabulary
        "MAX_LEN"                   : 15,
        "TOKENIZER_PATH"            : os.path.join(PROJECT_DIR, "..", "dataset/jeps_tokenizer.json"),
        "TOKEN_LEVEL"               : "W", # C: Character - W: Word
        "ACTION_DESC_MAX_LEN"       : 64,
        "CMD_MAX_LEN"               : 128,
        "VALIDATION_PCT"            : .25,
        "NUM_COMMANDS"              : 46,   # The number of motor commands in the dataset
        "TRAIN_TFMS"                : {
                "Resize"                  : {"height": IMAGE_SIZE,"width": IMAGE_SIZE},
                "RandomBrightnessContrast": {'p': 0.2},
        },
        "TEST_TFMS"        : {
                "Resize"                  : {"height": IMAGE_SIZE,"width": IMAGE_SIZE},
        }  
    }

    TRAIN = {
        "TRAIN_BATCH_SIZE"            : 16,    # Batch size to use when training the dataset
        "TEST_BATCH_SIZE"             : 8,    # Batch size to use when testing/validating the dataset
        "MAX_EPOCH"                   : 50, #500   # The maximum number epochs to train the model for
        "GPU"                         : True,  # Whether to use GPU or not when it is available
        "GPU_DEVICE"                  : "cuda:0",
        "NUM_WORKERS"                 : 4,
        "TF_RATE"                     : .99,
        "MAX_TF_RATE_STEPS"           : 4,
        "TF_RATE_DECAY"               : .05,   
        "N_WORKERS"                   : os.cpu_count(),
        "LEARNING_RATE"               : 1e-3,                 # The learning rate that will be used by optimizer(s)
        "MIN_LEARNING_RATE"           : 8e-7,                 
        "OPTIMIZER"                   : "Adam",     
    }


    MODEL = {
        "CNN_BACKBONES"      : {
        "resnet18"        : "ResNet18_Weights.IMAGENET1K_V1",
        "resnet34"        : "ResNet34_Weights.IMAGENET1K_V1",
        "resnet50"        : "ResNet50_Weights.IMAGENET1K_V1",
        "convnext_tiny"   : "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        "efficientnet_b4" : "EfficientNet_B4_Weights.IMAGENET1K_V1"

    },
    "CNN_FC_DIM"                  : 256,   
    "EMB_DROPOUT_RATE"            : .15,
    "N_LSTM_LAYERS_ENC"           : 2,
    "ENCODER_HIDDEN_DIM"          : 256,
    "DECODER_HIDDEN_DIM"          : 256,
    "DECODER_DROPOUT_RATE"        : .3,
    "KEY_VALUE_SIZE"              : 128, 
    }

    WANDB = {
            "PROJECT"               : "jepsam-s23",
            "GROUP"                 : "exp1",
            "USERNAME"              : "dric225",
            
            "CONFIG"                : {
                    'batch_size'    : TRAIN["TRAIN_BATCH_SIZE"],
                    'lr'            : TRAIN["LEARNING_RATE"],
                    'epochs'        : TRAIN["MAX_EPOCH"],
                    'ablation_name' : EXP_NAME,
                    # 'lr_scheduler': {
                    #     "patience": 5,
                    #     "gamma": 0.2
                    # }
                } 
    }


if __name__ == "__main__":

    pprint(vars(Config))