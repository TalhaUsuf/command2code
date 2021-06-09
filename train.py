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


def main(argv):
    Console().rule(title="[bold red]Training Script[/bold red]", style="red on black", align="center")

    with Console().status("Loading the yaml config file ...."):
        config = yaml.load(open("./lstm_torch_tokenizer/conf.yaml", "r"), yaml.SafeLoader)
        args = argparse.Namespace(**config)

    if FLAGS.t:
        with Console().status("Working with tfidf features", spinner="aesthetic"):
            features = joblib.load("tfidf_features.pkl")
            Console().log(f"tfidf features shape ---> {features.A.shape}")

            model = lstm_embedding_model(args)
            Console().print(model)







if __name__=='__main__':
    app.run(main)

