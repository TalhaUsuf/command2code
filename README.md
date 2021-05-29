# Introduction
This repository uses **stacked-LSTM** architecture using **two** layers to 
train on the given data.


> Images after the training are saved inside `images` folder.

> Uses `tensorflow tokenizer` for tokenization of text.

> Uses embedding for word vectorization. 
# Hyper Parameters
| Parameters | Values |
| :---------- | -------:|
batch | 12
classes| 6
dropout | 0.5
embed_dim | 128
epochs | 1500
hidden_1 | 1024
lstm_hidden | 8
max_norm | 1.0
num_directions | 1
num_layers | 2
vocab | 16

# Installaion
>`pip install -r requirements.txt`
# How To Train
>Set parameters inside `config.yaml` file, also set dataset path. 

> `python train.py` to start training

> Dataset files are located inside the `dataset` folder

# TODO
> Add prediction function which directly takes `text-command` from user.


# Results
|  |
|:-----:|
|Image shows `Epochs vs the F1-score` with size corresponding to the `loss` values| 
|![Epochs vs F1 score loss](images/epochs_f1_loss.png)|
|Image shows `Epochs vs Loss`| 
|![Epochs vs loss](images/epochs_loss.png)|
|Image shows `Epochs vs the Learning-Rate`|
|![Epochs vs LR](images/epochs_lr.png)|
|Image shows `Loss vs F1-score`|
|![Loss vs F1-score](images/loss_f1.png)|