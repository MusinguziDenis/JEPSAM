import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ae_resnet import get_configs, ResNetEncoder, ResNetDecoder

from config import Config as cfg




class JEPSAMEncder(nn.Module):
    def __init__(
            self,
            cnn_backbone_name,
            vocab_size,
            embedding_dim,
            hidden_dim,
            latent_dim
    ):
        super().__init__()
        # CNN ftr extractor
        configs, bottleneck = get_configs(cnn_backbone_name)

        self.image_feature_extractor = nn.Sequential(
            ResNetEncoder(configs, bottleneck),
            nn.Flatten(),
            nn.Linear(in_features=2048*7*7,
                      out_features=cfg.MODEL["CNN_FC_DIM"])
        )

        # Semantic Embedding
        self.embedding = SemanticEmbedding(vocab_size, embedding_dim)

        # Encoder
        self.action_encoder = nn.LSTM(embedding_dim, hidden_dim)

        # Episodic Memory
        self.episodic_memory = EpisodicEncoder(hidden_dim, hidden_dim)

        # Features mixer
        self.feature_mixer = nn.Linear(2*hidden_dim, latent_dim)

    def forward(self, x_action_desc, x_img):
        # Semantic Embedding & encoding
        semantic_embed = self.embedding(x_action_desc)
        semantic_out, (hidden, cell) = self.action_encoder(semantic_embed)

        # Episodic Memory
        embedded_state = self.image_feature_extractor(x_img)
        episodic_out, _ = self.episodic_memory(embedded_state)

        # Fusion
        concatenated_input = torch.cat((episodic_out, semantic_out), dim=1)
        encoder_output = self.feature_mixer(concatenated_input)

        return encoder_output, (hidden, cell)


class JEPSAMDecder(nn.Module):
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
        self.encoder = JEPSAMEncder(
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
    pass
