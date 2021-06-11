import joblib
import numpy as np
from rich.console import Console
from absl.flags import FLAGS
from absl import app, flags
import time
from lstm_torch_tokenizer.model import lstm_embedding_model
import argparse
import numpy as np
from rich.console import Console
import torch
import yaml
import torch
from sklearn.utils.class_weight import compute_class_weight
from torchmetrics import Accuracy, F1
from tqdm import trange
from rich.console import Console
from torch.optim import AdamW
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import shutil






flags.DEFINE_bool('t', default=False, help="if given ---> performs training on tfidf features")
flags.DEFINE_bool('k', default=False, help="if given ---> performs training on keras tokenized features")


def main(argv):
    Console().rule(title="[bold red]Training Script[/bold red]", style="red on black", align="center")

    with Console().status("Loading the yaml config file ...."):
        config = yaml.load(open("./lstm_torch_tokenizer/conf.yaml", "r"), yaml.SafeLoader)
        args = argparse.Namespace(**config)

        main_enc = joblib.load("enc_main.pkl")  # loading the encoders
        minor_enc = joblib.load("enc_sub.pkl")  # loading the minor encoders
        N_classes = len(main_enc.classes_)
        # update the cfg file accordingly
        args.classes = N_classes
        config["classes"] = N_classes # update the yaml file


    if FLAGS.t:
        with Console().status("Working with tfidf features .... ", spinner="aesthetic"):
            features = joblib.load("tfidf_features.pkl")

            Console().log(f"tfidf features shape ---> {features.A.shape}")



            model = lstm_embedding_model(args)
            Console().print(model)
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #             tfidf cannot be used with embedding + lstm
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if FLAGS.k:
#         perform training on keras feature-set
        with Console().status("Working with keras-tokenized-features ....", spinner="aesthetic"):
            features = joblib.load("kerasTok_features.pkl")
            Console().log("read the features file")
            tok = joblib.load("keras_tokenizer.pkl")
            vocab = len(tok.word_index) + 1
            args.vocab = vocab
            config["vocab"] = vocab
            Console().log("Updated vocabulary size in yaml file ...")
            yaml.dump(config, open("./lstm_torch_tokenizer/conf.yaml","w"), yaml.SafeDumper)

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #         For tfidf features, embedding-layer cannot
            #                be used hence using lstm_without_embedding_model
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





if __name__=='__main__':
    app.run(main)

