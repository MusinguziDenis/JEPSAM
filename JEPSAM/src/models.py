import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torchinfo import summary

from utils.ae_resnet import get_configs, ResNetEncoder, ResNetDecoder

from config import Config as cfg

from dataloader import get_dataloaders

import os.path as osp

import pandas as pd

import logging
logging.basicConfig(level="INFO")

class EpisodicEncoder(nn.Module):
    def __init__(
        self, 
        cnn_backbone_name:str="resnet50",
        hidden_dim:int=cfg.MODEL["CNN_FC_DIM"]
    ):
        super().__init__()
        
        configs, bottleneck = get_configs(cnn_backbone_name)
        
        self.feature_extractor = ResNetEncoder(configs, bottleneck)
        
        n_ftrs = 2048 * (cfg.IMAGE_SIZE // 32) * (cfg.IMAGE_SIZE // 32)
        self.projection_layer = nn.Linear(
            in_features=n_ftrs,
            out_features=hidden_dim
        )
        

    def forward(self, x_perceived):
        
        B, C, H, W = x_perceived.shape
        
        ftrs = self.feature_extractor(x_perceived)
        # print("ftrs: ", ftrs.shape)
        out = self.projection_layer(ftrs.view(B, -1))
        # print("out: ", out.shape)
        
        return out

class SemanticEncoder(nn.Module):
    def __init__(
        self, 
        vocab_size:int=cfg.DATASET["NUM_TOTAL_TOKENS"], 
        embedding_dim:int=cfg.MODEL["CNN_FC_DIM"],
        num_layers:int=2
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embedding_dim,
            # padding_idx=cfg.DATASET["PAD"]
        )
        
        self.action_encoder = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=embedding_dim // 2,
            num_layers=num_layers,
            bidirectional = True,
            batch_first=True
        ) 

    def forward(self, packed_ad, ad_lens):
        
        # Unpack the packed sequence
        input_data, batch_sizes, _, _ = packed_ad
        embedded = self.embedding(input_data)
        # Pack the embedded sequence back
        packed_embedded = pack_padded_sequence(embedded, ad_lens, enforce_sorted=False, batch_first=True)
        # Apply LSTM on the packed sequence
        packed_output, (h_n, c_n) = self.action_encoder(packed_embedded)
        # Unpack the LSTM output
        out, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        return out, h_n, c_n
    
class JEPSAMEncoder(nn.Module):
    def __init__(
            self,
            vocab_size:int=cfg.DATASET["NUM_TOTAL_TOKENS"], 
            embedding_dim:int=cfg.MODEL["CNN_FC_DIM"],
            num_layers:int=2
    ):
        super().__init__()
        
        # Semantic Encoder
        self.semantic_encoder = SemanticEncoder(
            vocab_size=vocab_size,
            num_layers=num_layers,
            embedding_dim=embedding_dim
        )

        # Episodic Encoder
        self.episodic_encoder = EpisodicEncoder(hidden_dim=embedding_dim)

        # Features mixer
        self.feature_mixer = nn.LSTM(
            input_size=embedding_dim*2, 
            hidden_size=embedding_dim,
            num_layers= num_layers,
            bidirectional = True,
            batch_first=True
        )

    def forward(self, x_ad, x_ad_lens, x_perceived):
        # pack padded inputs 
        packed = pack_padded_sequence(
            input=x_ad, 
            lengths= x_ad_lens, 
            batch_first=True, 
            enforce_sorted=False
        )
        # 1. Semantic Embedding & encoding
        semantic_enc, h_ad, c_ad = self.semantic_encoder(packed, x_ad_lens)
        _, seq_len, _ = semantic_enc.shape
        logging.info(f"Semantic encoding: \t\t{semantic_enc.shape}")
        
        # 2. Episodic encoding
        episodic_enc = self.episodic_encoder(x_perceived)
        logging.info(f"Episodic encoding: \t\t{episodic_enc.shape}")
        
        repeated_ep_emb = episodic_enc.unsqueeze(1).expand(
            episodic_enc.shape[0], 
            seq_len, 
            episodic_enc.shape[-1]
        )
        logging.info(f"Repeated embedding: \t\t{repeated_ep_emb.shape}")
        # 3. Fusion
        concat_ftrs = torch.cat((repeated_ep_emb, semantic_enc), dim=-1)
        logging.info(f"Concat ftrs: \t\t\t{concat_ftrs.shape}")

        fused_ftrs, (h_fused, c_fused) = self.feature_mixer(concat_ftrs)
        logging.info(f"Fused features: \t\t{fused_ftrs.shape}")

        return fused_ftrs, h_fused, c_fused, h_ad, c_ad


class JEPSAMDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        latent_dim,
        output_dim,
        motor_commands_dim
    ) -> None:
        super().__init__()
        # Attention Mechanism
        self.attention = Attention(hidden_dim, hidden_dim)
        # Image generator (CNN)
        self.image_generator = GeneratorCNN(latent_dim, output_dim)

        # Action generator (RNN)
        self.action_generator = ActionGenerationModule(latent_dim, output_dim)

    def forward(self, encoder_out):
        pass


class JEPSAM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        motor_commands_dim,
        latent_dim,
        cnn_backbone_name
    ):
        super().__init__()

        # Encoder
        self.encoder = JEPSAMEncoder(
            cnn_backbone_name,
            vocab_size,
            embedding_dim,
            hidden_dim
        )

        # Decoder
        self.decoder = JEPSAMDecder()

    def forward(self, action_desc, initial_state):
        # Semantic Embedding
        embedded_action = self.embedding(action_desc)
        embedded_state = self.embedding(initial_state)

        # Encoder
        encoder_output, (hidden, cell) = self.encoder_lstm(embedded_action)

        # Episodic Memory
        episodic_output, _ = self.episodic_memory(encoder_output)

        # Attention Mechanism
        attended_output = self.attention(
            episodic_output, hidden, encoder_output)

        # Generator (GAN)
        latent_repr = torch.randn(encoder_output.size(
            0), self.latent_dim).to(encoder_output.device)
        generated_image = self.generator(latent_repr)

        # Decoder
        decoder_input = torch.cat((embedded_state, attended_output), dim=2)
        decoder_output, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        goal_image = self.fc_goal(decoder_output)
        motor_commands = self.fc_motor(decoder_output)
        return goal_image, motor_commands


if __name__ == "__main__":
    logging.info("Checking data loading pipeline")
    tdf = pd.read_csv(
        osp.join(cfg.DATASET['PATH'], "v1/updated_train.csv")
    )

    train_dl, val_dl = get_dataloaders(
        train_df=tdf
    )

    # fetch example from dataloader
    for data in train_dl:
        in_state, goal_state, ad, cmd, ad_lens, cmd_lens = data[0], data[1], data[2], data[3], data[4], data[5]
        # print("In\t\t\t:", in_state.shape)
        # print("Action desc\t\t:", ad.shape)
        # print("Action desc (len)\t:", ad_lens.shape)    
        break

    # build encoder
    encoder = JEPSAMEncoder().to(cfg.TRAIN["GPU_DEVICE"])
    summary(model=encoder)

    # forward through encoder

    E, hn_E, cn_E, hn_ad, cn_ad = encoder(
        x_ad=ad.to(cfg.TRAIN["GPU_DEVICE"]), 
        x_ad_lens=ad_lens, 
        x_perceived=in_state.to(cfg.TRAIN["GPU_DEVICE"])
        )


    # build decoder
    # decoder = JEPSAMDecoder().to(cfg.TRAIN["GPU_DEVICE"])
    # summary(model=decoder)