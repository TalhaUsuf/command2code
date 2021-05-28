import argparse
import numpy as np
from rich.console import Console
import torch
import yaml
from model import lstm_embedding_model
import torch
from sklearn.utils.class_weight import compute_class_weight
from dataset import load_data
from tqdm import trange
from rich.console import Console
from torch.optim import AdamW
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

config = yaml.load(open("conf.yaml", "r"), yaml.SafeLoader)
args = argparse.Namespace(**config)

x, y, weights, sq_len, vocab, bs = load_data()
# loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights))
loss = torch.nn.CrossEntropyLoss()
args.vocab = vocab
args.batch = bs
x = torch.tensor(x).type(torch.LongTensor)
y = torch.tensor(y).type(torch.LongTensor)
sq_len = torch.tensor(sq_len).type(torch.long).cpu()

model = lstm_embedding_model(args)
optim = AdamW(params=model.parameters(),
              lr=0.001,

              )

n = 500
schedule = LinearWarmupCosineAnnealingLR(
    optimizer=optim,
    warmup_epochs=int((1 / 10) * n),
    max_epochs=n,
)
if __name__ == '__main__':
    print = Console().print

    for epoch in trange(n, desc="EPOCH", leave=False):
        # print(x)
        # print(y)
        # print(sq_len)
        #
        # print(type(sq_len))
        # break

        out = model(x, sq_len)
        # print(f"pred shape --> {out.shape}")
        # print(f"target shape --> {y.shape}")
        l = loss(out, y)
        # print(f"{'EPOCH':<15}{'--->':<25}{str(l.detach().numpy().squeeze())}")
        l.backward(retain_graph=True)
        optim.step()
        # schedule.step() # in epoch loop
        # optim.zero_grad()

if __name__ == '__main__':
    pass
