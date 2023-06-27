from addict import Dict

import copy

from einops import rearrange

import numpy as np

import pandas as pd

import os.path as osp

import torch
from torch.nn.functional import one_hot
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torchinfo import summary
import torchvision.models as torchvision_models

from typing import List, Tuple, Union

import sys

import models
from utils.parser import load_config, parse_args
import utils.model_utils as model_utils
from utils.ae_resnet import get_configs, ResNetEncoder, ResNetDecoder

from dataloader import get_dataloaders, SimpleTokenizer, JEPSAMDataset
import vocabulary as vocab

class JEPSAMEncoder(nn.Module):
    def __init__(
        self, 
        cfg: Dict, 
        cnn_backbone_name:str="resnet50"
    ):
        super().__init__()
        
        self.cfg = cfg
        
        self.device = self.cfg.TRAIN.GPU_DEVICE if torch.cuda.is_available() else "cpu"
        
        # embedding layer
        self.embedding = nn.Embedding(cfg.DATASET.VOCABULARY_SIZE, cfg.AEJEPS.EMBEDDING_DIM)        
        
        # CNN ftr extractor
        configs, bottleneck = get_configs(cnn_backbone_name)

        self.image_feature_extractor = nn.Sequential(
            ResNetEncoder(configs, bottleneck),
            nn.Flatten(),
            nn.Linear(in_features = 2048*7*7, out_features=cfg.AEJEPS.CNN_FC_DIM)
        )
        
        # features mixer
        self.feature_mixing = nn.LSTM(
            input_size=cfg.AEJEPS.EMBEDDING_DIM, 
            hidden_size=cfg.AEJEPS.HIDDEN_DIM, 
            num_layers=cfg.AEJEPS.NUM_LAYERS_ENCODER,
            dropout=cfg.AEJEPS.ENCODER_DROPOUT, 
            bidirectional=cfg.AEJEPS.IS_BIDIRECTIONAL
        )
                
        self.to(self.device)
        
        
    def forward(self, inp:Union[List, list, dict], mode:str='train'):
        """
        
        """
        if isinstance(inp, list) or isinstance(inp, List):
            in_state, goal_state, ad, cmd, ad_lens, cmd_lens = inp
        else:
            _, in_state, goal_state, ad, cmd = inp['sample_id'], inp['in_state'], inp['goal_state'], inp['action_desc'], inp["motor_cmd"]
            ad = ad["ids"]
            # print(ad["length"])
            ad_lens = ad["length"]
            
            cmd = cmd["ids"]
            cmd_lens = cmd["length"]
        
        in_state, goal_state, ad, cmd, ad_lens, cmd_lens = in_state.to(self.device), goal_state.to(self.device), ad.to(self.device), cmd.to(self.device), ad_lens.to(self.device), cmd_lens.to(self.device)
        # print(in_state.device)
        B, _, max_len = ad.shape

        # 1. Image feature extraction
        feats_per = self.image_feature_extractor(in_state)
        # feats_per = self.img_projection(feats_per.view(B, -1))
        # print(feats_per.shape)
        feats_per = feats_per.repeat((1, max_len)).reshape((B, max_len, -1))
        
        if mode =="train":
            feats_goal = self.image_feature_extractor(goal_state)
            # feats_goal = self.img_projection(feats_goal.view(B, -1))
            # print(feats_goal.shape)
            
            feats_goal = feats_goal.repeat((1, max_len)).reshape((B, max_len, -1))
        else:
            pass
        
        # print(f"feats_per: {feats_per.shape}")
        # print(f"feats_goal: {feats_goal.shape}")
        
        # 2. Text feature extraction
        action_desc_emb = self.embedding(ad)#.squeeze(1)
        
        if mode =="train":
            motor_cmd_emb = self.embedding(cmd)#.squeeze(1)
            # For each batch entry determine the length of the longest of the text sequence
            lengths_max = [max(ltext, lcmd)
                           for ltext, lcmd in zip(ad_lens, cmd_lens)]        
            # print(lengths_max)
        else:
            lengths_max = [ltext for ltext in ad_lens]     
        # 3. Feature Fusion
        # Optional: add a projection layer that will 
        # print(feats_per.shape, feats_goal.shape, action_desc_emb.shape, motor_cmd_emb.shape)
        
        # print( 
        #     feats_per.shape, 
        #     feats_goal.shape, 
        #     action_desc_emb.squeeze(1).shape, 
        #     motor_cmd_emb.squeeze(1).shape
        #      )
        concat_feats = torch.cat((
            feats_per, 
            feats_goal, 
            action_desc_emb.squeeze(1), 
            motor_cmd_emb.squeeze(1)
        ), dim=-2)#.squeeze(1)
        
        # print(f"Fused feats: {concat_feats.shape}")
        
        # 4. Feature mixing
        # packed_input = concat_feats
        packed_input = pack_padded_sequence(
            input=concat_feats, 
            lengths=lengths_max, 
            enforce_sorted=False, 
            batch_first=True
        )
        
        output, (hidden, carousel) = self.feature_mixing(packed_input)
        # print(output.shape)
        output, len_output = pad_packed_sequence(output, batch_first= True)
        
        return output, len_output, hidden, carousel

class JEPSAMDecoder(nn.Module):
    def __init__(
        self, 
        cfg: Dict,
        cnn_backbone_name:str="resnet50",
        
    ):
        super().__init__()
        
        # class attributes
        self.cfg = cfg
        self.cnn_backbone_name = cnn_backbone_name
        self.num_directions = 2 if cfg.AEJEPS.IS_BIDIRECTIONAL else 1
        decoder_hidden_dim = self.num_directions * cfg.AEJEPS.HIDDEN_DIM
        
        ## Layers
        # tokenizer 
        # self.tokenizer  = PreTrainedTokenizerFast(
        #     tokenizer_file= self.cfg.DATASET.TOKENIZER_PATH, # You can load from the tokenizer file, alternatively
        #     unk_token="[UNK]",
        #     pad_token="[PAD]",
        #     cls_token="[CLS]",
        #     sep_token="[SEP]",
        #     mask_token="[MASK]",
        # )
        
        self.tokenizer = SimpleTokenizer(vocab)

        # embedding layer - same as encoder embedding layer
        self.embedding = nn.Embedding(
            cfg.DATASET.VOCABULARY_SIZE, 
            cfg.AEJEPS.EMBEDDING_DIM,
            device=self.cfg.TRAIN.GPU_DEVICE
        )
        
        # image decoder
        configs, bottleneck = get_configs(cnn_backbone_name)
        self.img_projection = nn.Sequential(
            nn.Linear(in_features=decoder_hidden_dim, out_features=cfg.AEJEPS.CNN_FC_DIM ),
            getattr(nn, cfg.AEJEPS.DECODER_ACTIVATION)(),
            nn.Linear(in_features=cfg.AEJEPS.CNN_FC_DIM , out_features=2048*7*7),
        ).to(self.cfg.TRAIN.GPU_DEVICE)
        
        self.img_decoder = ResNetDecoder(configs[::-1], bottleneck).to(self.cfg.TRAIN.GPU_DEVICE)
        
        # motor command decoding layer
        self.motor_decoder = nn.LSTMCell(
            input_size=cfg.AEJEPS.EMBEDDING_DIM, 
            hidden_size=decoder_hidden_dim 
        ).to(self.cfg.TRAIN.GPU_DEVICE)

        self.activation_motor = getattr(nn, cfg.AEJEPS.ACTIVATION_MOTOR)()

        
        # action desc. decoding layer
        self.lang_decoder = nn.LSTMCell(
            input_size=cfg.AEJEPS.EMBEDDING_DIM, 
            hidden_size=decoder_hidden_dim
        ).to(self.cfg.TRAIN.GPU_DEVICE)

        self.activation_lang = getattr(nn, cfg.AEJEPS.ACTIVATION_LANG)()

        
        # projection layers
        # self.hidden_to_conv_in = nn.Linear(
        #     in_features=decoder_hidden_dim, 
        #     out_features=self.cfg.AEJEPS.HIDDEN_TO_CONV
        # )
            
        self.lang_head = nn.Linear(
            in_features=decoder_hidden_dim, 
            out_features=cfg.DATASET.VOCABULARY_SIZE 
            # To be discussed use the individual vocabs or the merged one for the projection
        ).to(self.cfg.TRAIN.GPU_DEVICE)

        self.motor_cmd_head = nn.Linear(
            in_features=decoder_hidden_dim, 
            out_features=cfg.DATASET.VOCABULARY_SIZE 
            # To be discussed use the individual vocabs or the merged one for the projection
        ).to(self.cfg.TRAIN.GPU_DEVICE)
        
        self.device = self.cfg.TRAIN.GPU_DEVICE if torch.cuda.is_available() else "cpu"
        
    def forward(
        self,
        enc_output, 
        len_enc_output, 
        hidden, 
        carousel,
        mode:str="train"
    ):
            
        batch_size, max_len, num_ftrs = enc_output.shape
        
        # hidden
        # hidden = hidden.view(self.num_directions, self.num_layers, batch_size, -1)
        # hidden = hidden[:self.num_directions, self.num_layers - 1, :, :]  # Take the last forward direction hidden state for
        
        hidden, carousel = self._rearrange_states(hidden, carousel)

        cmd_h_t, lang_h_t = (hidden, carousel), (hidden, carousel)
            
        # Unsqueeze to match expected input by transposed convolutions
        self.hidden = hidden.unsqueeze(0)
        
        # run decoding steps
        # generate action desc from latent representation
        lang_out = self._decode_action_description(hidden=lang_h_t, batch_size=batch_size, max_len=max_len)
        # generate motor cmd from latent representation
        motor_out = self._decode_motor_command(hidden=cmd_h_t, batch_size=batch_size, max_len=max_len)
        # reconstruct from latent representation
        per_image_rec = self._reconstruct_image(hidden)
        # generate from latent representation
        goal_image = self._generate_goal_image(hidden)
        
        return per_image_rec, goal_image, lang_out, motor_out
    
    
    def _rearrange_states(self, hidden, carousel):
        """
        
        """
        # hidden
        hidden = rearrange(
            hidden, 
            '(d l) b h -> l b (d h)',
            d=self.num_directions, 
            l=self.cfg.AEJEPS.NUM_LAYERS_ENCODER
        )
        hidden = hidden[self.cfg.AEJEPS.NUM_LAYERS_ENCODER - 1, :, :]
        
        # carousel
        carousel = rearrange(
            carousel, 
            '(d l) b h -> l b (d h)',
             d=self.num_directions, 
            l=self.cfg.AEJEPS.NUM_LAYERS_ENCODER
        )
        carousel = carousel[self.cfg.AEJEPS.NUM_LAYERS_ENCODER - 1, :, :]
        
        return hidden, carousel
    
    
    def _decode_action_description(
        self, 
        hidden, 
        batch_size:int, 
        max_len:int
    ):
        """
        """
        # print(next(self.lang_decoder.parameters()).is_cuda)
        
        lang_out = []        
        # Initialize the predictions with [SOS]
        prediction_txt_t = torch.ones(batch_size, 1).to(self.device).long() * self.cfg.DATASET.SOS
        # print(prediction_txt_t.device)
        for t in range(max_len):
            char = self.embedding(prediction_txt_t).squeeze(1)
            # hidden state at time step t for each RNN
            hidden, lang_c_t = self.lang_decoder(char, hidden)
            # project hidden state to vocab
            lang_scores = self.activation_lang(self.lang_head(hidden))
            # update hidden states
            hidden = (hidden, lang_c_t)
            # store newly generated token
            lang_out.append(lang_scores.unsqueeze(1))
            # draw new token: greedy decoding
            prediction_txt_t = lang_scores.argmax(dim=1)
        
        return torch.cat(lang_out, 1)
            
    def _decode_motor_command(
        self, 
        hidden, 
        batch_size:int, 
        max_len:int,
        method:str="embed"
    ):
        """
        Parameters:
        ----------
        
            method: str
                The method to use for token 
        """
        
        motor_out = []
        # Initialize the predictions with [SOS]
        prediction_cmd_t = torch.ones(batch_size, 1).to(self.device).long() * self.cfg.DATASET.SOS
            
        for t in range(max_len):
            if method=="one-hot":
                command = one_hot(
                    prediction_cmd_t.long(),
                    num_classes=cfg.DATASET.NUM_COMMANDS
                ).squeeze(1).float()
            else:
                command = self.embedding(prediction_cmd_t).squeeze(1)
            
            # hidden state at time step t for each RNN
            hidden, cmd_c_t = self.motor_decoder(command, hidden)
            # project hidden state to vocab
            cmd_scores = self.activation_motor(self.motor_cmd_head(hidden))
            # update hidden states
            hidden = (hidden, cmd_c_t)
            # store newly generated token
            motor_out.append(cmd_scores.unsqueeze(1))
            # draw new token: greedy decoding
            prediction_cmd_t = cmd_scores.argmax(dim=1)
            
        return torch.cat(motor_out, 1)

    def _reconstruct_image(self, hidden):
        """
        """
        conv_in = self.img_projection(hidden)
        # print(conv_in.shape)
        B, _ = conv_in.shape
        
        return self.img_decoder(conv_in.view(B, 2048, 7, 7))
    
    def _generate_goal_image(self, hidden):
        """
        """
        return self._reconstruct_image(hidden)# to be fixed
 
    def pred_to_str(
        self, 
        predictions:torch.Tensor
    )->list:
        """
            Decode predictions (from ids to token)
            
            Parameters:
            ----------
                - predictions: Tensor
                    batch predictions from decoder module
        """
        return self.tokenizer.batch_decode(predictions.argmax(dim=-1))
    
class JEPSAM(nn.Module):
    """
    This class is an Autoencoder based deep learning implementation of a Joint Episdoic, Procedural, and Semantic Memory.

    Parameters
    ----------

    """
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg

        # encoder
        self.encoder = JEPSAMEncoder(cfg=self.cfg)
        # decoder
        self.decoder = JEPSAMDecoder(cfg=self.cfg)
        # weight tying
        self.decoder.embedding.weight = self.encoder.embedding.weight
        
        self.device = self.cfg.TRAIN.GPU_DEVICE if torch.cuda.is_available() else "cpu"

        
    def forward(
        self, 
        inp, 
        mode:str="train"
    ):
        """
        """
        
        # encode
        o, lo, h, c  = self.encoder(inp)
        
        # decode
        reconstructed_image, goal_image, decoded_action_desc, decoded_cmd = self.decoder(
            enc_output=o, 
            len_enc_output=lo, 
            hidden=h, 
            carousel=c
        )

        return reconstructed_image, goal_image, decoded_action_desc, decoded_cmd




if __name__ == "__main__":

    cfg = load_config()

    tdf = pd.read_csv(
        osp.join(cfg.DATASET.PATH, "updated_train.csv")
    )

    tdf.head()
    train_dl, _ = get_dataloaders(
        train_df=tdf,
        cfg=cfg
    )

    for data in train_dl:
        pass
        break

    # Encoder
    # encoder = JEPSAMEncoder(
    #     cnn_backbone_name="resnet50",
    #     cfg=cfg
    # )

    # o, lo, h, c = encoder(data)

    # print("enc out: ", o.shape)

    # # model summary
    # summary(
    #     model=encoder,
    #     input_dict=data,
    #     col_names=["kernel_size", "num_params"],

    # )


    # Decoder
    # decoder = JEPSAMDecoder(cfg=cfg).to(cfg.TRAIN.GPU_DEVICE)
    # per_image_rec, goal_image, lang_out, motor_out = decoder(
    #     enc_output=o, 
    #     len_enc_output=lo, 
    #     hidden=h, 
    #     carousel=c
    # )
    # print(per_image_rec.shape, goal_image.shape, lang_out.shape, motor_out.shape)


    # JEPSAM
    jepsam = JEPSAM(cfg)
    print(jepsam)
    per_image_rec, goal_image, lang_out, motor_out = jepsam(data)
    print("output shapes: ", per_image_rec.shape, goal_image.shape, lang_out.shape, motor_out.shape)

