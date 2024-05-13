import os
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from Config import Config, Path


def data_generator(path_labels, path_dataset):
    config = Config()
    dir_annotation = os.listdir(path_labels)

    first = True # Used to mark whether the label data is loaded for the first time
    
    #stack all the labels
    for f in dir_annotation:
        if first:
            labels = np.load(os.path.join(path_labels, f))
            first = False
        else:
            temp = np.load(os.path.join(path_labels, f))
            labels = np.append(labels, temp, axis=0)
    # obtains the labels of all subjects 
    labels = torch.from_numpy(labels) #transformer the labels from numpy to tensor type
    
    dataset_EEG_FpzCz = np.load(os.path.join(path_dataset, 'TF_EEG_Fpz-Cz_mean_std.npy')).astype('float32')
    dataset_EEG_PzOz = np.load(os.path.join(path_dataset, 'TF_EEG_Pz-Oz_mean_std.npy')).astype('float32')
    dataset_EOG = np.load(os.path.join(path_dataset, 'TF_EOG_mean_std.npy')).astype('float32')
    
    print('TF_EEG_FpzCz', dataset_EEG_FpzCz.shape)
    '''remember to move MP files to TF_data'''
    MP_EEG_FpzCz = np.load(os.path.join(path_dataset, 'EEG_Fpz-Cz_interpld_MP.npy')).astype('float32')
    MP_EEG_PzOz = np.load(os.path.join(path_dataset, 'Pz-Oz_interpld_MP.npy')).astype('float32')
    MP_EOG = np.load(os.path.join(path_dataset, 'EOG_interpld_MP.npy')).astype('float32')
    
    mp_set = np.stack((MP_EEG_FpzCz, MP_EEG_PzOz, MP_EOG), axis = 1) # stack the mp data 
    mp_set = torch.from_numpy(mp_set)
    
    dataset = np.stack((dataset_EEG_FpzCz, dataset_EEG_PzOz, dataset_EOG), axis = 1) 
    dataset = torch.from_numpy(dataset)

    print('dataset: ', dataset.shape) #(196350, 3, 29, 128)
    print('mp_data:', mp_set.shape)
    
    
    # hold out the validation set
    X_train_test, X_val, y_train_test, y_val = train_test_split(dataset, labels, test_size=1/(config.num_fold+1), random_state=0, stratify=labels)
    
    MP_train_test, MP_val, _, _ = train_test_split(mp_set, labels, test_size=1/(config.num_fold+1), random_state=0, stratify=labels)


    val_set = TensorDataset(X_val, MP_val, y_val)  # 包括常规输入、Matrix Profile和标签

    val_loader = DataLoader(dataset=val_set, batch_size=config.batch_size, shuffle=False)

    print('val_set:', len(X_val))
    return X_train_test, MP_train_test, y_train_test, val_loader
