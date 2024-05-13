# MPSleepNet
Implementation of MPSleepNet, a robust deep learning model for precise classification of sleep stages using polysomnography data. 
## Abstract
See the whole architecture from the feature below:
<img width="239" alt="image" src="https://github.com/EliteriaLYU/MPSleepNet/assets/146659503/e670a9f8-65d5-4c94-b1cf-99afde3ebb86">

## Requirements
python = 3.9
ptroch = 1.9.0
thop
tqdm
numpy 
sklearn
scipy = 1.5.4
mne 

## Data
we used three public datasets in this study:
- SleepEDF-78
- SleepEDF-20
- SHHS

## Reproducibility 
If you want to update the hyperparameters of model, please edit the `Config.py`. Include but not limit to:
- Device (CPU or CUDA)
- Batchsize
- Number of folds: K
- The number of training epochs
- Normal hyperparameters: lr, dropout, the number of attention heads and transformer encoders etc.
Follow these steps to reproduce:
1. Run `dataset_prepare.py` to preprocess(i.e. annotaions, splits to epochs,  the raw data, 
