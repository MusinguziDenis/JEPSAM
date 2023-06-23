# Taken from the open source implementation of ResNet at
# https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html


from addict import Dict

import os.path as osp
import pandas as pd

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

import torchvision.models as torchvision_models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np
from einops import rearrange
import math
from torch.nn.functional import one_hot

from torchinfo import summary

from utils.parser import load_config
from utils.model_utils import freeze_module

from dataloader import get_dataloaders

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = x
        x = self.fc(x)

        return x, out

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


class AutoencoderJEPS(nn.Module):
    """
    This class is an Autoencoder based deep learning implementation of a Joint Episdoic, Procedural, and Semantic Memory.

    Parameters
    ----------
    cfg : Dict
        A Dict containing configuration settings

    Examples
    ---------
    >>> from utils.parser import load_config
    >>> from utils.data_utils import get_bacth
    >>> cfg_file = load_config('configs/default.yaml')
    >>> aejeps = AutoencoderJEPS(cfg_file)
    >>> batch = get_bacth(cfg_file)
    >>> # Using AEJEPS in train mode
    >>> goal_image, lang, cmd = aejeps(*batch, mode='train')
    >>> print(goal_image.shape, lang.shape, cmd.shape)
    >>> # Testing AEJEPS in mode text
    >>> goal_image, lang, cmd = aejeps(*batch, mode='text')
    >>> print(goal_image.shape, lang.shape, cmd.shape)
    >>> # Testing AEJEPS in mode command
    >>> goal_image, lang, cmd = aejeps(*batch, mode='command')
    >>> print(goal_image.shape, lang.shape, cmd.shape)

    """

    def __init__(self, cfg: Dict):
        super().__init__()
        #: The integer representing the start-of-sequence symbol
        self.sos = cfg.DATASET.SOS
        self.eos = cfg.DATASET.EOS
        # Get the parameters for the encoder RNN
        embedding_dim = cfg.AEJEPS.EMBEDDING_DIM
        hidden_dim = cfg.AEJEPS.HIDDEN_DIM
        num_layers_enc = cfg.AEJEPS.NUM_LAYERS_ENCODER
        batch_first = cfg.AEJEPS.BATCH_FIRST
        dropout_rate_enc = cfg.AEJEPS.DROPOUT_ENCODER
        bidirectional_enc = cfg.AEJEPS.BIDIRECTIONAL_ENCODER

        # Get the parameters for the embedding layer
        vocabulary_size = cfg.DATASET.VOCABULARY_SIZE

        # Get the parameters for the motor decoder RNN
        motor_dim = cfg.DATASET.NUM_COMMANDS
        num_layers_motor = cfg.AEJEPS.NUM_LAYERS_MOTOR

        # Get the parameters for the motor decoder RNN
        num_layers_lang = cfg.AEJEPS.NUM_LAYERS_LANG

        # Get the number of motor commands
        num_motor_commands = cfg.DATASET.NUM_COMMANDS

        # Get Image Size
        image_size = cfg.DATASET.IMAGE_SIZE

        # Save the following parameter for use in the forward funciton
        self.num_directions = 2 if bidirectional_enc else 1
        self.num_layers = num_layers_enc
        self.batch_first = batch_first

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)

        # Automatically load weights for ImageNet
        self.feature_extractor_cnn = resnet50(pretrained=True)

        encoder_input_dim = embedding_dim + 2 * \
            self.feature_extractor_cnn.fc.in_features + num_motor_commands

        self.encoder = nn.LSTM(encoder_input_dim, hidden_dim, num_layers_enc, batch_first=batch_first,
                               dropout=dropout_rate_enc, bidirectional=bidirectional_enc)

        decoder_hidden_dim = self.num_directions * hidden_dim
        self.motor_decoder = nn.LSTMCell(
            motor_dim, decoder_hidden_dim, num_layers_motor)

        self.lang_decoder = nn.LSTMCell(
            embedding_dim, decoder_hidden_dim, num_layers_lang)

        self.hidden_to_conv_in = nn.Linear(decoder_hidden_dim, 1024)
        self.lang_preds = nn.Linear(decoder_hidden_dim, vocabulary_size)
        self.mot_preds = nn.Linear(decoder_hidden_dim, num_motor_commands)

        self.hidden2img = self.__get_transposed_convs(
            decoder_hidden_dim, image_size)

        # Freeze CNN so it will not be trained
        freeze_module(self.feature_extractor_cnn)

    def __get_transposed_convs(self, decoder_hidden_dim, image_size):
        tconv1 = nn.ConvTranspose2d(1, 4, 3, 2, 3, 0)
        tconv2 = nn.ConvTranspose2d(4, 8, 5, 2, 3, 0)
        tconv3 = nn.ConvTranspose2d(8, 16, 7, 2, 4, 1)
        tconv4 = nn.ConvTranspose2d(16, 3, 11, 1, 7, 0)

        return nn.Sequential(tconv1, tconv2, tconv3, tconv4)

    def forward(self, per_img: torch.Tensor, goal_img: torch.Tensor, text: torch.Tensor, commands: torch.Tensor,
                lengths_text: list, lengths_cmd: list, mode: str = 'train') -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        This function takes a perceived image, goal image, text, motor command sequence, and the lengths of the text and
        motor command sequence to learn a joint hidden representation from all the modalities. The joint representation
        is then used to reconstruct the three input modalities in train mode. In generation mode two of the input modalities
         are used to generate the third one. In generation mode, the sequence that is being generated needs to be given
         as input. This is required because the encoder operations expect all the inputs they were trained on. Thus,
         to prove correct operation in generation mode a special trivial input with the required size can be given at
         test time. The trivial input may also be used at train time to teach the model to ignore the trivial input.

        Parameters
        ----------
        per_img : torch.Tensor
            A batch of perceived images of shape (batch_size, height, width, 3)
        goal_img : torch.Tensor
            A batch of goal images of shape (batch_size, height, width, 3)
        text : torch.Tensor
            A batch of padded integer-encoded text input of shape (batch_size, max_sequence_length)
        commands : torch.Tensor
            A batch of padded one-hot encoded motor command sequence input of shape(batch_size, max_sequence_length)
        lengths_text : list
            The lengths of the text sequences in the batch
        lengths_cmd : list
            The lengths of the motor command sequences in the batch
        mode : str
            A string indicating the mode the autoencoder will be used in can be one of 'train', 'image',
                'command', 'text'.<br/>
                'train': Train the autoencoder<br/>
                'image': Use the autoencoder to generate the goal image<br/>
                'command': Use the autoencoder to generate the sequence of motor commands<br/>
                'text': Use the autoencoder to generate the text description

        Returns
        -------
        goal_image_rec : torch.Tensor
            The reconstructed (generated) goal image of shape (batch_size, 3, 224, 224)
        text : torch.Tensor
            The reconstructed (generated) text sequence of shape (batch_size, max_sequence_length, vocabulary_size)
        command : torch.Tensor
            The reconstructed (generated) motor command sequence of shape (batch_size, max_sequence_length, num_motor_commands)
        """

        batch_size, max_len, *_ = text.shape
        num_commands = commands.shape[2]

        _, feats_per = self.feature_extractor_cnn(
            per_img)  # (batch_size, feat_dim)
        _, feats_goal = self.feature_extractor_cnn(goal_img)

        # (batch_size, max_len) -> (batch_size, max_len, embedding_dim)
        text = self.embedding(text.long())

        # For each batch entry determine the length of the longest of the text sequence
        lengths_max = [max(ltext, lcmd)
                       for ltext, lcmd in zip(lengths_text, lengths_cmd)]

        # Batch size x feat_dim -> Batch_size x (max_len x feat_dim) -> Batch_size x max_len x feat_dim
        feats_per = feats_per.repeat((1, max_len)).reshape(
            (batch_size, max_len, -1))
        feats_goal = feats_goal.repeat(
            (1, max_len)).reshape((batch_size, max_len, -1))

        # concatenate the features
        concat_feats = torch.cat(
            (feats_per, feats_goal, text, commands), dim=2)

        packed_input = pack_padded_sequence(
            concat_feats, lengths_max, enforce_sorted=False, batch_first=self.batch_first)
        
        output, (hidden, carousel) = self.encoder(packed_input)

        # hidden
        # hidden = hidden.view(self.num_directions, self.num_layers, batch_size, -1)
        # hidden = hidden[:self.num_directions, self.num_layers - 1, :, :]  # Take the last forward direction hidden state for
        hidden = rearrange(hidden, '(d l) b h -> l b (d h)',
                           d=self.num_directions, l=self.num_layers)
        hidden = hidden[self.num_layers - 1, :, :]

        carousel = rearrange(carousel, '(d l) b h -> l b (d h)',
                             d=self.num_directions, l=self.num_layers)
        carousel = carousel[self.num_layers - 1, :, :]

        cmd_h_t, lang_h_t = (hidden, carousel), (hidden, carousel)
        # Unsqueeze to match expected input by transposed convolutions
        hidden = hidden.unsqueeze(0)

        motor_out = []
        lang_out = []
        device = per_img.device  # All tensors must live in the same device

        # Initialize the predictions of the two decoders RNNs at time step t to <sos> value
        prediction_cmd_t = torch.ones(
            batch_size, 1).to(device).long() * self.sos
        prediction_txt_t = torch.ones(
            batch_size, 1).to(device).long() * self.sos
        for t in range(max_len):
            # If in train mode use actual inputs, if in generation mode use prediction
            if mode == 'train':
                command = commands[:, t, :]
                char = text[:, t, :]
            elif mode == 'command':
                command = one_hot(prediction_cmd_t.long(),
                                  num_classes=num_commands).squeeze(1).float()
                char = text[:, t, :]
            elif mode == 'text':
                command = commands[:, t, :]
                char = self.embedding(prediction_txt_t).squeeze(1)

            # hidden state at time step t for each RNN
            cmd_h_t, cmd_c_t = self.motor_decoder(command, cmd_h_t)
            lang_h_t, lang_c_t = self.lang_decoder(char, lang_h_t)

            cmd_scores = self.mot_preds(cmd_h_t)
            lang_scores = self.lang_preds(lang_h_t)

            cmd_h_t = (cmd_h_t, cmd_c_t)
            lang_h_t = (lang_h_t, lang_c_t)

            motor_out.append(cmd_scores.unsqueeze(1))
            lang_out.append(lang_scores.unsqueeze(1))

            prediction_cmd_t = cmd_scores.argmax(dim=1)
            prediction_txt_t = lang_scores.argmax(dim=1)

        # hidden = hidden[:, :, :self.sqrt_dim ** 2]
        conv_in = self.hidden_to_conv_in(hidden)
        conv_in = rearrange(conv_in, 'l b (h1 h2) -> b l h1 h2', h1=32, h2=32)
        goal_image_rec = self.hidden2img(conv_in)
        return goal_image_rec, torch.cat(lang_out, 1), torch.cat(motor_out, 1)

## Summer 2023

def get_cnn_backbone(
    cfg:Dict, 
    backbone_name:str="resnet50", 
    freeze:bool=True,
    fc_out:int=None
):
    backbone = getattr(torchvision_models, backbone_name)(weights=cfg.MODEL.CNN_BACKBONES[backbone_name])
    
    # freeze backbone if specified
    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False
    
    if fc_out is not None:
        # resnet-based models
        if "resnet" in backbone_name.lower():
            backbone.fc = nn.Linear(in_features=backbone.fc.in_features, out_features=fc_out)

        if "convnext" in backbone_name.lower():
            pass
            
        if "densenet" in backbone_name.lower():
            pass

    return backbone

class JEPSAMEncoder(nn.Module):
    def __init__(
        self, 
        cfg: Dict, 
        cnn_backbone_name:str="resnet50", 
        cnn_fc_out:int=512
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(cfg.DATASET.VOCABULARY_SIZE, cfg.AEJEPS.EMBEDDING_DIM)
        
        self.image_feature_extractor = get_cnn_backbone(
            cfg=cfg,
            backbone_name=cnn_backbone_name,
            fc_out=cnn_fc_out
        )
        
        # features mixer
        encoder_input_dim = cfg.AEJEPS.EMBEDDING_DIM + 2 * self.image_feature_extractor.fc.in_features + cfg.DATASET.NUM_COMMANDS

        self.feature_mixing = nn.LSTM(
            input_size=cfg.AEJEPS.EMBEDDING_DIM, 
            hidden_size=cfg.AEJEPS.HIDDEN_DIM, 
            num_layers=cfg.AEJEPS.NUM_LAYERS_ENCODER,
            dropout=cfg.AEJEPS.ENCODER_DROPOUT, 
            bidirectional=cfg.AEJEPS.IS_BIDIRECTIONAL
        )
        
    
    def forward(self, inp:dict, mode:str='train'):
        """
        
        """
        B, _, max_len = inp["action_desc"]["ids"].shape

        # 1. Image feature extraction
        feats_per = self.image_feature_extractor(inp["in_state"])
        feats_per = feats_per.repeat((1, max_len)).reshape((B, max_len, -1))
        
        if mode =="train":
            feats_goal = self.image_feature_extractor(inp["goal_state"])
            feats_goal = feats_goal.repeat((1, max_len)).reshape((B, max_len, -1))
        else:
            pass
        
        # print(f"feats_per: {feats_per.shape}")
        # print(f"feats_goal: {feats_goal.shape}")
        
        # 2. Text feature extraction
        action_desc_emb = self.embedding(inp["action_desc"]["ids"])#.squeeze(1)
        
        if mode =="train":
            motor_cmd_emb = self.embedding(inp["motor_cmd"]["ids"])#.squeeze(1)
            # For each batch entry determine the length of the longest of the text sequence
            lengths_max = [max(ltext, lcmd)
                           for ltext, lcmd in zip(inp["action_desc"]["length"], inp["motor_cmd"]["length"])]
        else:
            lengths_max = [ltext for ltext in inp["action_desc"]["length"]]     
        # 3. Feature Fusion
        # Optional: add a projection layer that will 
        # print(feats_per.shape, feats_goal.shape, action_desc_emb.shape, motor_cmd_emb.shape)
        
        concat_feats = torch.cat((
            feats_per.unsqueeze(1), 
            feats_goal.unsqueeze(1), 
            action_desc_emb, 
            motor_cmd_emb
        ), dim=2).squeeze(1)
        
        print(f"Fused feats: {concat_feats.shape}")
        
        # 4. Feature mixing
        packed_input = pack_padded_sequence(
            input=concat_feats, 
            lengths=lengths_max, 
            enforce_sorted=False, 
            batch_first=True
        )
        
        output, (hidden, carousel) = self.feature_mixing(packed_input)
        
        output, len_output = pad_packed_sequence(output, batch_first= True)
        
        return output, len_output, hidden, carousel
    

if __name__ == "__main__":

    cfg = load_config()

    # aejeps = AutoencoderJEPS(cfg)
    # print(aejeps)

    encoder = JEPSAMEncoder(
    cnn_backbone_name="resnet18",
        cfg=cfg, 
        cnn_fc_out = cfg.AEJEPS.HIDDEN_DIM
    )

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


    o, lo, h, c = encoder(data)

    print(o.shape)

    # model summary
    summary(
        model=encoder,
        input_dict=data,
        col_names=["kernel_size", "num_params"],

    )

    # print("Testing AEJEPS in mode text")
    # goal_image, lang, cmd = aejeps(*batch, mode='train')
    # print(goal_image.shape, lang.shape, cmd.shape)

    # print("Testing AEJEPS in mode text")
    # goal_image, lang, cmd = aejeps(*batch, mode='text')
    # print(goal_image.shape, lang.shape, cmd.shape)

    # print("Testing AEJEPS in mode command")
    # goal_image, lang, cmd = aejeps(*batch, mode='command')
    # print(goal_image.shape, lang.shape, cmd.shape)
