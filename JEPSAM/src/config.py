import os

import torch as th

class Config:

    # I/O params
    data_dir                    = "../../dataset/"
    model_zoo                   = "../models/"
    log_dir                     = "../logs/"

    # Vocab 
    LETTER_LIST = [
        "[PAD]", "[SOS]", "[EOS]", 
        'A',   'B',    'C',    'D',    
        'E',   'F',    'G',    'H',    
        'I',   'J',    'K',    'L',       
        'M',   'N',    'O',    'P',    
        'Q',   'R',    'S',    'T', 
        'U',   'V',    'W',    'X', 
        'Y',   'Z',    "'",    ' ', 
    ]
    PAD_VALUE = LETTER_LIST.index("[PAD]")
    special_tokens = ["[SOS]", "[EOS]", "[PAD]"]
    vocab_size = len(LETTER_LIST)    
    
    # Training params
    lr                          = 1e-3
    min_lr                      = 1e-6
    weight_decay                = 1e-6
    optimizer                   = "Adam"
    train_batch_size            = 16
    test_batch_size             = 16
    max_epochs                  = 20
    device                      = "cuda:0"

    # Model params
    cnn_out                     = 256
    n_workers                   = os.cpu_count()
    emb_dropout_rate            = .15
    n_lstm_enc_layers           = 2
    encoder_hidden_dim          = 256
    decoder_hidden_dim          = 512
    decoder_dropout_rate             = .3
    key_value_size              = 128

