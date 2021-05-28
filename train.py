import argparse
import numpy as np
from rich.console import Console
import torch
import yaml
from model import lstm_embedding_model
import torch
from sklearn.utils.class_weight import compute_class_weight
from torchmetrics import Accuracy, F1
from dataset import load_data
from tqdm import trange
from rich.console import Console
from torch.optim import AdamW
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

Path("images").mkdir(parents=True, exist_ok=True)
if any(Path("images").iterdir()): # if files are present inside "images"
    shutil.rmtree("images") # then delete the whole tree 
    Path("images").mkdir(exist_ok=True, parents=True)

config = yaml.load(open("conf.yaml", "r"), yaml.SafeLoader)
args = argparse.Namespace(**config)

x, y, weights, sq_len, vocab, bs = load_data()
args.vocab = vocab
args.batch = bs
yaml.dump(vars(args), open("conf.yaml", "w"), yaml.SafeDumper)
# loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights))
loss = torch.nn.CrossEntropyLoss()
args.vocab = vocab
args.batch = bs

x = torch.tensor(x).type(torch.LongTensor)
y = torch.tensor(y).type(torch.LongTensor)
sq_len = torch.tensor(sq_len).type(torch.long).cpu()

model = lstm_embedding_model(args)
optim = AdamW(params=model.parameters(),
              lr=0.005,

              )

n = args.epochs
schedule = LinearWarmupCosineAnnealingLR(
    optimizer=optim,
    warmup_epochs=int((1 / 4) * n),
    max_epochs=n,
)
metric = F1(num_classes=args.classes,
            average="macro")

if __name__ == '__main__':
    print = Console().print

    logs = {"loss":[],
            "f1":[],
            "epochs":[],
            "lr":[]}
    for epoch in trange(n, desc="EPOCH", leave=False, position=0):
        logs["epochs"].append(epoch)
        # print(x)
        # print(y)
        # print(sq_len)
        #
        # print(type(sq_len))
        # break

        optim.zero_grad()
        out = model(x, sq_len)
        # print(f"pred shape --> {out.shape}")
        # print(f"target shape --> {y.shape}")
        l = loss(out, y)
        logs["loss"].append(l.detach().numpy().squeeze().tolist())
        f1 = metric(out, y)
        logs["f1"].append(f1)
        Console().print(f"EPOCH {epoch} ---> Loss {l.detach().numpy().squeeze()} F1 {f1:.4f} ---> LR {schedule.get_lr()[0]:.10f}")
        logs["lr"].append(schedule.get_lr()[0])

        # print(f"{f'EPOCH_{epoch}':<15}{'--->':<15}{l.detach().numpy().squeeze():.4f}{'':<15}{schedule.get_lr()[0]:.10f}")
        l.backward(retain_graph=True)
        optim.step()
        schedule.step() # in epoch loop

    logs = pd.DataFrame.from_dict(logs)
    sns.set_style("whitegrid")
    
    ax = sns.relplot(x="epochs",
                 y="lr",
                 data=logs,
                kind="line"
                 )
    ax.despine()
    ax.savefig("images/epochs_lr.png")
    ay = sns.relplot(x="epochs",
                 y="loss",
                 data=logs,
                kind="line"
                 )
    ay.despine()
    ay.savefig("images/epochs_loss.png")
    az = sns.relplot(x="epochs",
                     y="f1",
                     data=logs,
                     size="loss"
                     )
    az.despine()
    az.savefig("images/epochs_f1_loss.png")

    azz = sns.relplot(x="loss",
                     y="f1",
                     data=logs,
                     )
    azz.despine()
    azz.savefig("images/loss_f1.png")
    plt.show()