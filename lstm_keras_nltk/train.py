import torch
from pytorch_lightning import LightningModule, LightningDataModule
import torch.nn as nn
from argparse import Namespace
import numpy as np
from rich.console import Console
from rich.table import Table
from multiprocessing import cpu_count
import numpy as np
import json
import torch.nn.functional as F
from pytorch_lightning.metrics import functional as FM
from pytorch_lightning import Trainer
from torch.utils.data import random_split, DataLoader, Dataset
from pathlib import Path
import os
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger


class make_data(Dataset):
    """
    Gets x, y and tokenizes them using pre-trained BERT tokenizer
    """

    def __init__(self, x, y, tok, y_enc):
        """
        Takes in `x` and `y` as numpy arrays

        Parameters
        ----------
        x : np.ndarray
            shape --> [N, D]
        y : np.ndarray
            shape --> [N, D]
        """
        super(make_data, self).__init__()
        self.tokenizer = tok
        self.LENGTH = x.shape[0]
        # x, y, ------> arrays
        self.x = self.tokenizer.batch_encode_plus(list(x), truncation=True, padding=True)
        self.lengths = self.tokenizer.batch_encode_plus(list(x), truncation=True, padding=False)
        # self.y = self.tokenizer(text=list(y.squeeze()), truncation=True, padding=True)
        self.y = np.array([y_enc[k] for k in list(y)]).reshape(-1, 1)
        # Console().print(self.y, style="yellow")
        # Console().print(y, style="yellow")

    def __len__(self):
        """
        getter method to get total length of dataset
        """
        return self.LENGTH

    def __getitem__(self, item):
        """

        """
        x = self.x['input_ids'][item]
        y = self.y[item]
        L = [len(k) for k in self.lengths['input_ids']][item]

        return {'data': torch.tensor(x),
                'label': torch.squeeze(torch.from_numpy(y)).type(torch.LongTensor),
                'lengths': L}


class data_module(LightningDataModule):
    def __init__(self, make_data, batch_sz, CSV_file, test_sz_split, tok):
        """

        Parameters
        ----------
        make_data : subclass of Dataset
        """
        super().__init__()
        self.tokenizer = tok
        self.test_sz = test_sz_split
        self.make_data = make_data
        self.batch_sz = batch_sz
        self.csv = CSV_file

    def prepare_data(self):
        """
        Read the csv file. It is called on just ist gpu
        """
        data = pd.read_csv(str(self.csv), skipinitialspace=True)
        self.X, self.Y = data['Commands'].values, data['TargetFile'].values
        self.y_dict = {i: j for j, i in enumerate(set(list(self.Y)))}

    def setup(self, stage=None):
        """
        read full csv file and split the data according to test_sz. It is called on all GPUs
        """
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.X, self.Y, test_size=self.test_sz)

    def train_dataloader(self):
        self.train_dataset = make_data(x=self.xtrain, y=self.ytrain, tok=self.tokenizer, y_enc=self.y_dict)
        return DataLoader(self.train_dataset, batch_size=self.batch_sz, shuffle=True, num_workers=cpu_count())

    def val_dataloader(self):
        self.test_dataset = make_data(self.xtest, self.ytest, self.tokenizer, y_enc=self.y_dict)
        return DataLoader(self.test_dataset, batch_size=self.batch_sz, shuffle=False, num_workers=cpu_count())


class text2command(LightningModule):
    def __init__(self, params):
        tab = Table(title="text2command")
        tab.add_column("PARAM.", style="bold cyan", justify="right")
        tab.add_column("VALUE", style="bold yellow", justify="left")

        super(text2command, self).__init__()
        # Console().print(params, style="bold yellow")
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #                      save hparams for logging
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.hparams = params
        # self.log_hyperparams(self.hparams)
        self.save_hyperparameters(self.hparams)
        # parse the namespace parameters and cvt into dictionary
        params = vars(params)
        # Console().print(params, style="bold cyan")
        for k, v in params.items():
            tab.add_row(k, str(v))
        Console().print(tab)
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #                      layers
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.embed = nn.Embedding(params['vocab'], params['embed_dim'])
        self.lstm = nn.LSTM(input_size=params['embed_dim'], hidden_size=params['hidden'], num_layers=params['n_layers'],
                            batch_first=True)
        self.fc1 = nn.Linear(in_features=params['hidden'], out_features=params['h1'])
        self.fc2 = nn.Linear(in_features=params['h1'], out_features=params['h2'])
        self.fc3 = nn.Linear(in_features=params['h2'], out_features=params['nC'])
        self.nC = params['nC']
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #                      for lr-scheduler
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.step_size = params['step_sz']
        self.gamma = params['gamma']

    def forward(self, inp):
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #                      only used in inference
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        out = self.embed(inp)  # shape: [N, seq, embed_dim

        out, (h_n, c_n) = self.lstm(out)  # shape: [N, seq, hidden], [N, n_layers, hidden], [N, n_layers, hidden]
        out = out[:, -1, :]  # [N, hidden] last time step hidden value

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        prediction = F.softmax(self.fc3(out), dim=-1)

        return {'out': prediction}

    def training_step(self, batch, batch_idx):
        x, y, sq_len = batch['data'], batch['label'], batch['lengths']
        # Console().print(f"[green]{type(x)}\t\t{x.shape}[/green]")
        # Console().print(f"[green]{type(y)}\t\t{y.shape}[/green]")
        #         x shape ---> [N, D]
        out = self.embed(x)  # shape: [N, seq, embed_dim
        out = torch.nn.utils.rnn.pack_padded_sequence(out, lengths=sq_len, batch_first=True, enforce_sorted=False)
        out, (h_n, c_n) = self.lstm(out)  # shape: [N, seq, hidden], [N, n_layers, hidden], [N, n_layers, hidden]
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = out[:, -1, :]  # [N, hidden] last time step hidden value
        # [batch, seq_len, hidden] ---> O
        # [layers, batch, hidden] ---> hn
        # [layers, batch, hidden] ---> cn
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        pred = F.softmax(self.fc3(out), dim=-1)

        print(pred.shape, y.shape)
        print(f"PRED [red]{str(pred.shape)}[/red]")
        print(f"LABEL [green]{str(y.shape)}[/green]")
        loss = F.cross_entropy(pred, y)
        acc = FM.accuracy(torch.argmax(pred, dim=-1), y)

        self.log(name="train_loss", value=loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log(name="train_acc", value=acc, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return {'loss': loss, 'acc': acc}  # needed for internally aggregating metrics at epoch end and at step end

    def validation_step(self, batch, batch_idx):
        x, y, sq_len = batch['data'], batch['label'], batch['lengths']
        # Console().print(f"[green]{type(x)}\t\t{x.shape}[/green]")
        # Console().print(f"[green]{type(y)}\t\t{y.shape}[/green]")
        # Console().print(f"[green]{y}[/green]")
        #         x shappe ---> [N, D]
        out = self.embed(x)  # shape: [N, seq, embed_dim
        out = torch.nn.utils.rnn.pack_padded_sequence(out, lengths=sq_len, batch_first=True, enforce_sorted=False)
        out, (h_n, c_n) = self.lstm(out)  # shape: [N, seq, hidden], [N, n_layers, hidden], [N, n_layers, hidden]
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = out[:, -1, :]  # [N, hidden] last time step hidden value

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        pred = F.softmax(self.fc3(out), dim=-1)

        loss = F.cross_entropy(pred, y)

        acc = FM.accuracy(torch.argmax(pred, dim=-1), y)
        f1_score = FM.f1_score(torch.argmax(pred, dim=-1), y, num_classes=self.nC)

        self.log(name="val_loss", value=loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log(name="val_acc", value=acc, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log(name="val_F1", value=f1_score, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return {'loss': loss, 'acc': acc,
                'f1': f1_score}  # needed for internally aggregating metrics at epoch end and at step end

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters())
        scheduler = StepLR(optim, step_size=self.step_size, gamma=self.gamma)
        return [optim], [scheduler]


if __name__ == '__main__':
    print = Console().print
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cback_lr = LearningRateMonitor(logging_interval="step")
    cback_model = ModelCheckpoint(monitor='val_F1', mode='max',
                                  filepath='./logs/command2code-{epoch:02d}-{val_F1:.2f}', save_last=True,
                                  save_top_k=5)

    # logger = NeptuneLogger(api_key='ANONYMOUS',
    #                        project_name='shared/pytorch-lightning-integration',
    #                        experiment_name='command2code',  # Optional,
    #                        tags=['pytorch-lightning', 'mlp']  # Optional, ).log_hyperparams()
    #                        )
    vocab_sz = 28996  # from dm.tokenizer
    model = text2command(
        Namespace(embed_dim=128, hidden=32, n_layers=1, vocab=vocab_sz, h1=256, h2=16, nC=6, step_sz=2, gamma=0.01))

    dm = data_module(make_data, batch_sz=10, CSV_file='./command2code.csv', test_sz_split=0.10,
                     tok=BertTokenizerFast.from_pretrained('bert-base-cased'))
    #
    # trainer = Trainer(max_epochs=100, gpus=1, callbacks=[cback], logger=logger)

    trainer = Trainer(max_epochs=10, gpus=0, callbacks=[cback_lr], checkpoint_callback=cback_model)
    trainer.fit(model=model, datamodule=dm)


m = model.load_from_checkpoint("logs\command2code-epoch=04-val_F1=0.50.ckpt")