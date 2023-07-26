import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ae_resnet import get_configs, ResNetEncoder, ResNetDecoder

from config import Config as cfg


"""
            Action Description --> Semantic Embedding --> Encoder (LSTM) --> v1
                                        |                         |
                                        |                         |
            Initial State/Image --> Episodic Embedding --> Encoder (Linear or LSTM) --> v2

                                    -------- concat ----------
                                        v = (v1, v2)    
                                                |
                                                |
                                        Feature mixing (fusion)
                                                |
                                                |
                                                E (Encoder out)
                                                |                        
                                                |                        
                                        ------------------  
                                        | Attention Module |
                                        ------------------   
                                                |
                                                |
                                            ctx, attn_w (context vector, attn weights)
                                                |                        
                                                |     
                        ------------------         ------------------
                        | Image Generator |        | Action generator|
                        ------------------         ------------------
                                |                         |
                                |                         |
                            Goal Image               Motor Commands
                                
"""

class SemanticEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SemanticEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_text):
        embedded_text = self.embedding(input_text)
        return embedded_text

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.W_query = nn.Linear(input_dim, hidden_dim)
        self.W_key = nn.Linear(input_dim, hidden_dim)
        self.W_value = nn.Linear(input_dim, hidden_dim)
        self.output_dim = input_dim

    def forward(self, query, key, value):
        Q = self.W_query(query)
        K = self.W_key(key)
        V = self.W_value(value)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention_scores = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_scores, V)
        return output

class EpisodicEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)

    def forward(self, input_data):
        output, (hidden, cell) = self.lstm(input_data)
        return output, (hidden, cell)
    
class GeneratorCNN(nn.Module):
    def __init__(self, latent_dim, output_channels):
        super(GeneratorCNN, self).__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels

        # CNN layers for the generator
        self.conv1 = nn.Conv2d(latent_dim, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, latent_repr):
        # Reshape latent representation to a 4D tensor (batch_size, channels, height, width)
        latent_repr = latent_repr.view(latent_repr.size(0), self.latent_dim, 1, 1)

        # Pass the latent representation through the CNN layers
        x = self.relu(self.conv1(latent_repr))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        generated_image = self.conv4(x)

        return generated_image
    

class ImageGenerationModule(nn.Module):
    def __init__(
            self, 
            input_channels,
            vocab_size, 
            embedding_dim, 
            hidden_dim, 
            output_dim, 
            motor_commands_dim, 
            latent_dim
        ):
        super().__init__()

        # Attrs
        self.latent_dim = latent_dim

        # Semantic Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Encoder
        self.encoder_lstm = nn.LSTM(embedding_dim + input_channels, hidden_dim)

        # Episodic Memory (if applicable)
        self.episodic_memory = EpisodicMemory(hidden_dim, hidden_dim)

        # Attention Mechanism (if applicable)
        self.attention = Attention(hidden_dim, hidden_dim)

        # Decoder
        self.generator = GeneratorCNN(latent_dim, output_dim)

    def forward(self, action_desc, initial_state):
        # 1. Encode
        # Semantic Embedding
        embedded_action = self.embedding(action_desc)

        concatenated_input = torch.cat((embedded_action, initial_state), dim=1)

        # Encoder
        encoder_output, (hidden, cell) = self.encoder_lstm(concatenated_input)

        # Episodic Memory
        episodic_output, _ = self.episodic_memory(encoder_output)

        # Attention Mechanism
        attended_output = self.attention(episodic_output, hidden, encoder_output)

        # 2. Decode
        latent_repr = torch.randn(encoder_output.size(0), self.latent_dim).to(encoder_output.device)
        generated_image = self.generator(latent_repr)

        return generated_image

class ActionGenerationModule(nn.Module):
    def __init__(
            self,
            latent_dim, 
            output_dim
    ) -> None:
        super().__init__()

    def forward(self):
        pass

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
            nn.Linear(in_features=2048*7*7, out_features=cfg.MODEL["CNN_FC_DIM"])
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
        attended_output = self.attention(episodic_output, hidden, encoder_output)

        # Generator (GAN)
        latent_repr = torch.randn(encoder_output.size(0), self.latent_dim).to(encoder_output.device)
        generated_image = self.generator(latent_repr)

        # Decoder
        decoder_input = torch.cat((embedded_state, attended_output), dim=2)
        decoder_output, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        goal_image = self.fc_goal(decoder_output)
        motor_commands = self.fc_motor(decoder_output)
        return goal_image, motor_commands



if __name__ == "__main__":
    pass