

import torch
class Config(object):
    """args in model and trainer"""
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_fold = 10
        self.num_classes = 5
        self.num_epochs = 200               # Because early stopping is used, this parameter can be relatively large
        self.batch_size = 64
        self.pad_size = 29                  # time dimension of TF image
        self.learning_rate = 5e-6
        self.dropout = 0.1                  # dropout rate in transformer encoder
        self.dim_model = 128                # frequency of TF image
        self.forward_hidden = 1024          # hidden units of transformer encoder
        self.fc_hidden = 1024               # hidden units of FC layers
        self.num_head = 8
        self.num_encoder = 8               # number of encoders in single-channel feature extraction block
        self.num_encoder_multi = 4          # number of encoders in multi-channel feature fusion block

import os

class Path(object):
    """path of files in this project"""
    def __init__(self):
        
        self.path_PSG = './sleep_data/raw'
        self.path_hypnogram = './sleep_data/hypnogram'
        self.path_raw_data = './sleep_data/npy'
        self.path_labels = './sleep_data/npy/labels'
        self.path_TF = './sleep_data/npy/TF_data'
        self.path_MP = './MP_edf/'

        
        if not os.path.exists(self.path_hypnogram):
            os.makedirs(self.path_hypnogram)

        if not os.path.exists(self.path_raw_data):
            os.makedirs(self.path_raw_data)

        if not os.path.exists(self.path_TF):
            os.makedirs(self.path_TF)

