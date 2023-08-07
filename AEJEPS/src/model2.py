from addict import Dict

import copy

from einops import rearrange

import numpy as np

import pandas as pd

import os.path as osp

import torch
from torch.nn.functional import one_hot
import torch.nn as nn
import torch.nn.functional as F
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


class MyLSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout= 0, num_layers= 1):
        super().__init__()

        if num_layers == 1:
            lstms   = torch.nn.LSTMCell(input_size, output_size)
        else:
            lstms   = torch.nn.Sequential()
            lstms.append(torch.nn.LSTMCell(input_size, hidden_size))
            for i in range(num_layers-2):
                lstms.append(torch.nn.LSTMCell(hidden_size, hidden_size))
            lstms.append(torch.nn.LSTMCell(hidden_size, output_size))  

        self.lstm_cells     = lstms
        self.lstm_dropout   = torch.nn.Sequential(*[torch.nn.Dropout(dropout)]*num_layers)

    def __len__(self):
        return len(self.lstm_cells)

    def lstm_step(self, lstm_input, hidden_state):
        # hidden_state is a list
        for i in range(len(self.lstm_cells)):
            # print(hidden_state[i].shape)
            hidden_state[i] = self.lstm_cells[i](lstm_input, hidden_state[i])
            lstm_input      = hidden_state[i][0]
            lstm_input      = self.lstm_dropout[i](lstm_input)
        return lstm_input, hidden_state
    
    def forward(self, lstm_input, hidden_state):
        return self.lstm_step(lstm_input, hidden_state)
    
    
class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hidden_dim * 2)+ (dec_hidden_dim) , dec_hidden_dim)
        self.v    = nn.Linear(dec_hidden_dim, 1, bias =False)
        
    def forward(self, hidden, encoder_outputs):
        # print(hidden.shape)
        # print(encoder_outputs.shape)
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[1]
        
        # hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim =-1)
        
        # hidden = hidden.permute(1,0,2)
        
        # print("hidden_attn shape", hidden.shape)
        
        hidden = hidden.permute(1, 0, 2).repeat(1,src_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim =2)))
                
        #repeat decoder hidden state src_len times
        # hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        # energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        
        attention = F.softmax(attention, dim=1)
        
        #print("attention shape", attention.shape)
        
        
        return attention
        
    

class JEPSAMEncoder(nn.Module):
    def __init__(
        self,
        cfg: Dict,
        cnn_backbone_name: str = "resnet50"
    ):
        super().__init__()

        self.cfg = cfg

        self.device = self.cfg.TRAIN.GPU_DEVICE if torch.cuda.is_available() else "cpu"

        # embedding layer
        self.embedding = nn.Embedding(
            cfg.DATASET.VOCABULARY_SIZE, cfg.AEJEPS.EMBEDDING_DIM)

        # CNN ftr extractor
        configs, bottleneck = get_configs(cnn_backbone_name)

        self.image_feature_extractor = nn.Sequential(
            ResNetEncoder(configs, bottleneck),
            nn.Flatten(),
            nn.Linear(in_features=2048*7*7, out_features=cfg.AEJEPS.CNN_FC_DIM)
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

    def forward(self, inp: Union[List, list, dict], mode: str = 'train'):
        """

        """
        if isinstance(inp, list) or isinstance(inp, List):
            in_state, goal_state, ad, cmd, ad_lens, cmd_lens = inp
        else:
            _, in_state, goal_state, ad, cmd = inp['sample_id'], inp[
                'in_state'], inp['goal_state'], inp['action_desc'], inp["motor_cmd"]
            ad = ad["ids"]
            # print(ad["length"])
            ad_lens = ad["length"]

            cmd = cmd["ids"]
            cmd_lens = cmd["length"]

        in_state, goal_state, ad, cmd, ad_lens, cmd_lens = in_state.to(self.device), goal_state.to(
            self.device), ad.to(self.device), cmd.to(self.device), ad_lens.to(self.device), cmd_lens.to(self.device)
        # print(in_state.device)
        B, _, max_len = ad.shape

        # 1. Image feature extraction
        feats_per = self.image_feature_extractor(in_state)
        
        feats_goal = self.image_feature_extractor(goal_state)
        # feats_per = self.img_projection(feats_per.view(B, -1))
        # print(feats_per.shape)
        # feats_per = feats_per.repeat((1, max_len)).reshape((B, max_len, -1))
        
        # Add the lengths of the motor command and the action description
        total_len = ad_lens + cmd_lens
        
        # print("ad shape", ad.shape)
        # print("cmd shape", cmd.shape)
        
        # Concatenate the action descriptions and the motor commands
        concatenated_feats = torch.cat((cmd.squeeze(), ad.squeeze()), dim=1)
        
        # Embed the action descriptions and motor commands
        embed_concat_feats = self.embedding(concatenated_feats)
        
        # Concatenate the images, action descriptions and the motor command embeddings     
        combined_feats = torch.cat((embed_concat_feats, feats_per.unsqueeze(1), feats_goal.unsqueeze(1)), dim=1)
        
        # Increase the total length by 2
        total_len +=2
        total_len = total_len.to("cpu")
        
        # Pack the concatenated features
        packed_features = pack_padded_sequence(combined_feats, total_len,batch_first=True, enforce_sorted=False)
        
        # pass the images through the feature mixing layer
        output, (hidden, carousel) = self.feature_mixing(packed_features)
        
        # Unpack the features
        output, len_output = pad_packed_sequence(output, batch_first=True)

        return output, len_output, hidden, carousel        


class JEPSAMDecoder(nn.Module):
    def __init__(
        self,
        cfg: Dict,
        attention =None,
        cnn_backbone_name: str = "resnet50",
    ):
        super().__init__()

        # class attributes
        self.cfg = cfg
        self.cnn_backbone_name = cnn_backbone_name
        self.num_directions = 2 if cfg.AEJEPS.IS_BIDIRECTIONAL else 1
        decoder_hidden_dim = self.num_directions * cfg.AEJEPS.HIDDEN_DIM
        self.tf_ratio = self.cfg.TRAIN.TF_RATE
        self.action_rnn = nn.LSTM((self.cfg.AEJEPS.HIDDEN_DIM * 2) + self.cfg.AEJEPS.EMBEDDING_DIM, self.cfg.AEJEPS.HIDDEN_DIM)
        self.action_fc = nn.Linear((self.cfg.AEJEPS.HIDDEN_DIM * 2) + self.cfg.AEJEPS.EMBEDDING_DIM + self.cfg.AEJEPS.HIDDEN_DIM, self.cfg.DATASET.VOCABULARY_SIZE)

        # Layers
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
            nn.Linear(in_features=decoder_hidden_dim,
                      out_features=cfg.AEJEPS.CNN_FC_DIM),
            getattr(nn, cfg.AEJEPS.DECODER_ACTIVATION)(),
            nn.Linear(in_features=cfg.AEJEPS.CNN_FC_DIM,
                      out_features=2048*7*7),
        ).to(self.cfg.TRAIN.GPU_DEVICE)

        self.img_decoder = ResNetDecoder(
            configs[::-1], bottleneck).to(self.cfg.TRAIN.GPU_DEVICE)

        # motor command decoding layer
        self.motor_decoder = nn.LSTMCell(
            input_size=cfg.AEJEPS.EMBEDDING_DIM,
            hidden_size=decoder_hidden_dim
        ).to(self.cfg.TRAIN.GPU_DEVICE)
        
        self.command_decoder = MyLSTMCell(input_size = cfg.AEJEPS.EMBEDDING_DIM, 
                                           hidden_size = decoder_hidden_dim, 
                                           output_size = decoder_hidden_dim,
                                           dropout= 0, 
                                           num_layers= cfg.AEJEPS.NUM_LAYERS_MOTOR)

        self.activation_motor = getattr(nn, cfg.AEJEPS.ACTIVATION_MOTOR)()

        # action desc. decoding layer
        self.language_decoder = MyLSTMCell(input_size = cfg.AEJEPS.EMBEDDING_DIM, 
                                           hidden_size = decoder_hidden_dim, 
                                           output_size = decoder_hidden_dim,
                                           dropout= 0, 
                                           num_layers= cfg.AEJEPS.NUM_LAYERS_LANG)
        
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
        
        self.hidden_proj = nn.Linear(self.cfg.AEJEPS.HIDDEN_DIM *2, self.cfg.AEJEPS.HIDDEN_DIM)
        
        self.attention = attention

        self.device = self.cfg.TRAIN.GPU_DEVICE if torch.cuda.is_available() else "cpu"

    def forward(
        self,
        enc_output,
        len_enc_output,
        hidden,
        carousel,
        y_ad = None,
        y_cmd = None,
        mode: str = "train"
    ):
        # print(hidden.shape) # (n layers * n directions, batch_size, hidden_dim )
        # print(enc_output.shape) # (batch_size, seq_len, enc hid dim * 2)

        batch_size, max_len, num_ftrs = enc_output.shape

        # hidden
        # hidden = hidden.view(self.num_directions, self.num_layers, batch_size, -1)
        # hidden = hidden[:self.num_directions, self.num_layers - 1, :, :]  # Take the last forward direction hidden state for

        hidden, carousel, hidden_new, carousel_new = self._rearrange_states(hidden, carousel)
        
        # print(hidden.shape)
        lang_hidden_input = self.activation_lang(self.hidden_proj(hidden))
        # print(lang_hidden_input.shape)
        

        cmd_h_t, lang_h_t = (hidden, carousel), (hidden, carousel)
        
        # print("Cmd h_t shape",cmd_h_t[0].shape)
        
        # Repeat the language h_t to match the number of LSTM cells
        lang_h_t = [lang_h_t for i in range(self.cfg.AEJEPS.NUM_LAYERS_LANG)]
        cmd_h_t  = [cmd_h_t for i in range(self.cfg.AEJEPS.NUM_LAYERS_MOTOR)]

        # Unsqueeze to match expected input by transposed convolutions
        self.hidden = hidden.unsqueeze(0)
        
        language_output = self.decode_action_description(hidden_new, carousel_new, enc_output, batch_size,max_len, y = y_ad)

        command_output = self.decode_motor_command(hidden_new, carousel_new, enc_output, batch_size,max_len, y = y_cmd)


        # run decoding steps
        # generate action desc from latent representation
        lang_out = self._decode_action_description(
            hidden=lang_h_t, batch_size=batch_size, max_len=max_len, y = y_ad)
        # generate motor cmd from latent representation
        motor_out = self._decode_motor_command(
            hidden=cmd_h_t, batch_size=batch_size, max_len=max_len, y = y_cmd)
        # reconstruct from latent representation
        per_image_rec = self._reconstruct_image(hidden)
        # generate from latent representation
        goal_image = self._generate_goal_image(hidden)

        return per_image_rec, goal_image, lang_out, motor_out, language_output, command_output

    def _rearrange_states(self, hidden, carousel):
        """

        """
        # hidden
        # hidden = rearrange(
        #     hidden,
        #     '(d l) b h -> l b (d h)',
        #     d=self.num_directions,
        #     l=self.cfg.AEJEPS.NUM_LAYERS_ENCODER
        # )
        # hidden = hidden[self.cfg.AEJEPS.NUM_LAYERS_ENCODER - 1, :, :]
        hidden_new = hidden # Keep all the layers because that's the input to the new langauage decoder
        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        hidden     =  torch.cat((hidden_fwd, hidden_bwd), dim =-1)

        # carousel
        # carousel = rearrange(
        #     carousel,
        #     '(d l) b h -> l b (d h)',
        #     d=self.num_directions,
        #     l=self.cfg.AEJEPS.NUM_LAYERS_ENCODER
        # )
        # carousel = carousel[self.cfg.AEJEPS.NUM_LAYERS_ENCODER - 1, :, :]
        carousel_new = carousel
        carousel_fwd = carousel[-2, :, :]
        carousel_bwd = carousel[-1, :, :]
        carousel     = torch.cat((carousel_fwd, carousel_bwd), dim =-1)
        

        return hidden, carousel, hidden_new, carousel_new
    
    def decode_action_description(
        self,
        hidden,
        carousel,
        encoder_outputs,
        batch_size,
        max_len,
        y = None  
    ):
        if y is not None:
            max_len = y.shape[-1]

        lang_out = []
        # Initialize the predictions with [SOS]
        prediction_txt_t = torch.ones(1, batch_size).to(
            self.device).long() * self.cfg.DATASET.SOS
        
        # print("Hidden new shape", hidden.shape)
        
        # hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=-1).unsqueeze(0)
        
        # carousel = torch.cat((carousel[-2,:,:], carousel[-1,:,:]), dim=-1).unsqueeze(0)
        
        # h_t, c_t = self.hidden_proj(hidden), self.hidden_proj(carousel)
        h_t, c_t = hidden[-2,:,:].unsqueeze(0), carousel
        
        # print(y)
        # prediction_txt_t = [1, batch_size]
        for t in range(max_len):
            if t > 0 and y is not None:
                p = np.random.uniform(0,1)
                if (p<= self.tf_ratio):
                    char = self.embedding(y[:,:, t-1]).permute(1, 0, 2)
                # print(p)
            else:
                char = self.embedding(prediction_txt_t)#.permute(1, 0, 2)
                
                
            #char = [1, batch_size, embed_dim]
            # print("Char shape", char.shape)
            # print("hidden", hidden.shape)
            
            # print("h_t shape", h_t.shape)
                
            # a = self.attention(hidden, encoder_outputs)
            a = self.attention(h_t, encoder_outputs)
            
            #a =[batch_size, src_len]
            
            a = a.unsqueeze(1)
            
            #a =[batch_size, 1, src_len]
            
            # encoder_outputs = encoder_outputs.permute(1, 0, 2)
            
            #encoder_outputs = [batch_size, src_len, enc hid dim * 2]
            
            weighted = torch.bmm(a, encoder_outputs)
            
            #weighted = [batch_size, 1, enc hid dim * 2]
            
            weighted  = weighted.permute(1, 0, 2)
            
            #weighted = [1, batch_size, enc hid dim * 2]
            # print("weighted shape", weighted.shape)
            # print("Char shape", char.shape)
            
            if not self.training:
                # print("char shape", char.shape)
                # print("weighted shape", weighted.shape)
                if char.dim() < 3:
                    char = char.unsqueeze(0)
                # print("shapes")
                # print("Char shapes", char.shape)
                # print("Weighted shapes",weighted.shape)
                
                
            rnn_input = torch.cat((char, weighted), dim =2)
            
            #rnn_input = [1, batch_size, (enc hid dim * 2) + emb dim]
            
            # print("Hidden shape",hidden.unsqueeze(0).shape)
            # print("Rnn input shape", rnn_input.shape)
            
            # output, hidden = self.action_rnn(rnn_input, hidden.unsqueeze(0))
            
            output, (h_t, c_t) = self.action_rnn(rnn_input, (h_t[-1,:,:].unsqueeze(0), c_t[-1,:,:].unsqueeze(0)))
            
            #output = [seq len, batch_size, dec hid dim * n directions]
            #hidden[0], hidden[1] = [n layers * n directions, batch size, dec hid dim]
            
            char = char.squeeze(0)
            output   = output.squeeze(0)
            weighted = weighted.squeeze(0)
            
            prediction = self.activation_motor(self.action_fc(torch.cat((output, weighted, char), dim =1)))
            
            #prediction = [batch_size, output_dim]
            
            lang_out.append(prediction.unsqueeze(1))
            
            prediction_txt_t = prediction.argmax(dim=1)
            
        output = torch.cat(lang_out, 1)
        # print("Output shape",output.shape)
        
        return output
    
    def decode_motor_command(
        self,
        hidden,
        carousel,
        encoder_outputs,
        batch_size,
        max_len,
        y = None
    ):
        if y is not None:
            max_len = y.shape[-1]
            
        cmd_out = []
        # Initialize the predictions with [SOS]
        prediction_txt_t = torch.ones(1, batch_size).to(
            self.device).long() * self.cfg.DATASET.SOS
        
        h_t, c_t = hidden[-2,:,:].unsqueeze(0), carousel
        
        for t in range(max_len):
            if t > 0 and y is not None:
                p = np.random.uniform(0,1)
                if (p<= self.tf_ratio):
                    char = self.embedding(y[:,:, t-1]).permute(1, 0, 2)
                # print(p)
            else:
                char = self.embedding(prediction_txt_t)#.permute(1, 0, 2)
                
            a = self.attention(h_t, encoder_outputs)
            
            a = a.unsqueeze(1)
            
            weighted = torch.bmm(a, encoder_outputs)
            
            weighted  = weighted.permute(1, 0, 2)
            
            if not self.training:               
                if char.dim() < 3:
                    char = char.unsqueeze(0)
                    
            rnn_input = torch.cat((char, weighted), dim =2)
            
            output, (h_t, c_t) = self.action_rnn(rnn_input, (h_t[-1,:,:].unsqueeze(0), c_t[-1,:,:].unsqueeze(0)))

            char = char.squeeze(0)
            output   = output.squeeze(0)
            weighted = weighted.squeeze(0)
            
            prediction = self.activation_motor(self.action_fc(torch.cat((output, weighted, char), dim =1)))
            
            cmd_out.append(prediction.unsqueeze(1))

            prediction_txt_t = prediction.argmax(dim=1)

        output = torch.cat(cmd_out, 1)
        # print("Output shape",output.shape)
        
        return output
                      

    def _decode_action_description(
        self,
        hidden,
        batch_size: int,
        max_len: int,
        y = None
    ):
        """
        """
        # print(next(self.lang_decoder.parameters()).is_cuda)
        
        if y is not None:
            max_len = y.shape[-1]

        lang_out = []
        # Initialize the predictions with [SOS]
        prediction_txt_t = torch.ones(batch_size, 1).to(
            self.device).long() * self.cfg.DATASET.SOS
        
        lang_c_t   = hidden
        for t in range(max_len):
            if t > 0 and y is not None:
                p = np.random.uniform(0,1)
                if (p<= self.tf_ratio):
                    char = self.embedding(y[:,:, t-1]).squeeze(1)
            else:
                char = self.embedding(prediction_txt_t).squeeze(1)
            # hidden state at time step t for each RNN
            hidden, lang_c_t = self.language_decoder(char, lang_c_t)
            # project hidden state to vocab
            lang_scores = self.activation_lang(self.lang_head(hidden)) #changed the argument from hidden 
            # update hidden states
            # hidden = (hidden, lang_c_t)
            # hidden = lang_c_t
            # store newly generated token
            lang_out.append(lang_scores.unsqueeze(1))
            # draw new token: greedy decoding
            prediction_txt_t = lang_scores.argmax(dim=1)
            # print(prediction_txt_t)

        return torch.cat(lang_out, 1)

    def _decode_motor_command(
        self,
        hidden,
        batch_size: int,
        max_len: int,
        y = None,
        method: str = "embed"
    ):
        """
        Parameters:
        ----------

            method: str
                The method to use for token 
        """
        
        if y is not None:
            max_len = y.shape[-1]

        motor_out = []
        # Initialize the predictions with [SOS]
        prediction_cmd_t = torch.ones(batch_size, 1).to(
            self.device).long() * self.cfg.DATASET.SOS

        cmd_c_t = hidden
        
        for t in range(max_len):
            if method == "one-hot":
                command = one_hot(
                    prediction_cmd_t.long(),
                    num_classes=cfg.DATASET.NUM_COMMANDS
                ).squeeze(1).float()
            
                
            if y is not None and t > 0:
                p = np.random.uniform(0, 1)
                if(p <= self.tf_ratio):
                    command = self.embedding(y[:, :, t-1]).squeeze(1)
                else:
                    command = self.embedding(prediction_cmd_t).squeeze(1)
                    
            else:
                command = self.embedding(prediction_cmd_t).squeeze(1)

            # hidden state at time step t for each RNN
            hidden, cmd_c_t = self.command_decoder(command, cmd_c_t)
            # project hidden state to vocab
            cmd_scores = self.activation_motor(self.motor_cmd_head(hidden))
            # update hidden states
            # hidden = (hidden, cmd_c_t)
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
        return self._reconstruct_image(hidden)  # to be fixed

    def pred_to_str(
        self,
        predictions: torch.Tensor
    ) -> list:
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
    This class is an Autoencoder based deep learning implementation of a Joint Episdoic, Procedural, and Semantic Asociative Memory.

    Parameters
    ----------

    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg

        # encoder
        self.encoder = JEPSAMEncoder(cfg=self.cfg)
        # attention
        self.attention = Attention(self.cfg.AEJEPS.HIDDEN_DIM, self.cfg.AEJEPS.HIDDEN_DIM)
        # decoder
        self.decoder = JEPSAMDecoder(cfg=self.cfg, attention = self.attention)
        # weight tying
        self.decoder.embedding.weight = self.encoder.embedding.weight

        self.device = self.cfg.TRAIN.GPU_DEVICE if torch.cuda.is_available() else "cpu"

    def forward(
        self,
        inp,
        mode: str = "train"
    ):
        """
        """

        # encode
        o, lo, h, c = self.encoder(inp, mode=mode)

        # decode
        
        if mode == "train":
            _, _, ad, cmd, _, _ = inp
            
            reconstructed_image, goal_image, decoded_action_desc, decoded_cmd, language_output, command_output = self.decoder(
                enc_output=o,
                len_enc_output=lo,
                hidden=h,
                carousel=c,
                y_ad = ad.to(self.device),
                y_cmd = cmd.to(self.device), 
                mode=mode
            )
            
        else:
            reconstructed_image, goal_image, decoded_action_desc, decoded_cmd, language_output, command_output = self.decoder(
                enc_output=o,
                len_enc_output=lo,
                hidden=h,
                carousel=c,
                y_ad = None,
                y_cmd = None,
                mode=mode
            )

        return reconstructed_image, goal_image, decoded_action_desc, decoded_cmd, language_output


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
    print("output shapes: ", per_image_rec.shape,
          goal_image.shape, lang_out.shape, motor_out.shape)
