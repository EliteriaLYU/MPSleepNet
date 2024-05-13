import math
import copy
import torch
import collections
import numpy as np
import torch.nn as nn
from torch import Tensor
from copy import deepcopy
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import TransformerEncoderLayer, MultiheadAttention, Dropout, Linear, LayerNorm, Module, ModuleList,BatchNorm1d

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation) :
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model=128, dropout=0.1, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # pe:[1, 30, 128]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    '''
    MultiHeadAttention -> Add & Norm -> Feed forward -> Add & Norm
    '''
    def __init__(self, config, attn, layer_norm_eps=1e-6, activation='relu'):
        '''
        :param size: d_model
        :param attn:  Initialized MultiHeadAttention
        :param feed_forward: Initialized FFN
        :param dropout: Dropout rate
        '''
        super(EncoderLayer, self).__init__()
        self.attn = attn
        # self.feed_forward = feed_forward
        # Implementation of Feedforward model
        self.linear1 = Linear(config.dim_model , config.forward_hidden)
        self.dropout = Dropout(config.dropout)
        self.linear2 = Linear(config.forward_hidden , config.dim_model)

        self.norm1 = LayerNorm(config.dim_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(config.dim_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(config.dropout)
        self.dropout2 = Dropout(config.dropout)

        self.activation = _get_activation_fn(activation)


    def forward(self, x_a : Tensor, x_b: Tensor):
        '''
        :param x_a:  TF_data of each channel
        :param x_b:  matrix_profile of each channel
        :param mask: masked sa or not
        :return:
        '''
        src2 = self.attn(x_b, x_b, x_a)[0]
        src = x_a + self.dropout1(src2)
        src = self.norm1(src) #[64, 29, 128]


        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))  # (64, 29, 1024)

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class CustomEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):

        super(CustomEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        #self.norm = norm

    def forward(self, x_a: Tensor, x_b: Tensor)-> Tensor:
        '''
        :param x_a:
        :param x_b:
        :param src_mask:
        :return:
        '''
        for mod in self.layers:
            output = mod(x_a, x_b)

        return output


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        self.position_single = PositionalEncoding(d_model=config.dim_model, dropout=0.1)

        multi_attn = nn.MultiheadAttention(embed_dim=config.dim_model, num_heads=config.num_head, dropout=config.dropout, batch_first=True)

        custom_encoder_layer = EncoderLayer(config, multi_attn, layer_norm_eps=1e-6, activation='relu')
        self.transformer_encoder_1 = CustomEncoder(custom_encoder_layer, num_layers=config.num_encoder)
        self.transformer_encoder_2 = CustomEncoder(custom_encoder_layer, num_layers=config.num_encoder)
        self.transformer_encoder_3 = CustomEncoder(custom_encoder_layer, num_layers=config.num_encoder)



        self.drop = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(config.dim_model * 3)

        self.position_multi = PositionalEncoding(d_model=config.dim_model * 3, dropout=0.1)
        encoder_layer_multi = nn.TransformerEncoderLayer(d_model=config.dim_model * 3, nhead=config.num_head,
                                                         dim_feedforward=config.forward_hidden, dropout=config.dropout)
        self.transformer_encoder_multi = nn.TransformerEncoder(encoder_layer_multi, num_layers=config.num_encoder_multi)

        self.fc1 = nn.Sequential(
            nn.Linear(config.pad_size * config.dim_model * 3, config.fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(config.fc_hidden, config.num_classes)
        )

    def forward(self, x_a: Tensor, x_b: Tensor) -> Tensor:
        '''
        :param x_a: TF_data
        :param x_b: matrix_profile
        '''
        Xa_1 = x_a[:, 0, :, :]
        Xa_2 = x_a[:, 1, :, :]
        Xa_3 = x_a[:, 2, :, :]

        Xb_1 = x_b[:, 0, :, :]
        Xb_2 = x_b[:, 1, :, :]
        Xb_3 = x_b[:, 2, :, :]

        Xa_1 = self.position_single(Xa_1)
        Xa_2 = self.position_single(Xa_2)
        Xa_3 = self.position_single(Xa_3)

        Xb_1 = self.position_single(Xb_1)
        Xb_2 = self.position_single(Xb_2)
        Xb_3 = self.position_single(Xb_3)

        x1 = self.transformer_encoder_1(Xa_1, Xb_1)  # (batch_size, 29, 128)
        x2 = self.transformer_encoder_2(Xa_2, Xb_2)
        x3 = self.transformer_encoder_3(Xa_3, Xb_3)

        x = torch.cat([x1, x2, x3], dim=2)

        x = self.drop(x)
        x = self.layer_norm(x)
        residual = x

        x = self.position_multi(x)
        x = self.transformer_encoder_multi(x)

        x = self.layer_norm(x + residual)  # residual connection

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x