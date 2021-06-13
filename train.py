import joblib
import numpy as np
from rich.console import Console
import shutil
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
from torchmetrics import Accuracy, F1, ConfusionMatrix
from tqdm import trange
from rich.console import Console
from torch.optim import AdamW
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from lstm_without_embedding.model import lstm_without_embedding_model, save_ckpt
from tqdm import trange

sns.set_theme(style="darkgrid")


# Defining flags

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
        with Console().status("Working with tfidf features .... \n\n", spinner="aesthetic"):
            features = joblib.load("tfidf_features.pkl")
            enc = joblib.load("enc_main.pkl")
            labels = joblib.load("main_labels.pkl")
            labels_enc=torch.from_numpy(enc.transform(labels))
            Console().print(labels_enc)
            # print(labels_enc)
            print(35*"%")
            Console().log(f"tfidf features shape ---> {features.A.shape}")
            args.embed_dim = features.A.shape[-1]
            config["embed_dim"] = features.A.shape[-1]
            config["batch"] = features.A.shape[0]
            args.batch = int(features.A.shape[0])
            features = torch.from_numpy(features.A).type(torch.float32)
            yaml.dump(config, open("lstm_torch_tokenizer/conf.yaml", "w"), yaml.SafeDumper)
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #         For tfidf features, embedding-layer cannot
            #                be used hence using lstm_without_embedding_model
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            model = lstm_without_embedding_model(args)
            logs = {}
            n = args.epochs
            opt = torch.optim.AdamW(params=model.parameters(), lr=0.001)
            schedule = LinearWarmupCosineAnnealingLR(
                optimizer=opt,
                warmup_epochs=int((1 / 4) * n),
                max_epochs=n,
            )
            metric = F1(num_classes=args.classes,
                        average="macro")
            conf_mat = ConfusionMatrix( num_classes=args.classes)
            compute_loss = torch.nn.CrossEntropyLoss()
            prev_loss = 0
            for epoch in trange(args.epochs, desc="Epoch:"):
                # Console().log(f"input shape ---> {features.shape}")

                # Console().log(f"input shape ---> {model(features).shape}")
                opt.zero_grad()
                out = model(features)
                # Console().print(f"[cyan]{labels_enc.shape}")
                # Console().print(f"[cyan]{out.shape}")
                # print(35*"%")
                loss = compute_loss(out, labels_enc)
                logs.setdefault("loss", []).append(loss)
                f1 = metric(out, labels_enc)
                mat = conf_mat(out, labels_enc)
                logs.setdefault("conf_mat", []).append(mat)
                logs.setdefault("f1", []).append(f1)
                logs.setdefault("epoch", []).append(epoch)
                logs.setdefault("lr", []).append(schedule.get_lr()[0])

                if epoch == 0:
                    prev_loss = loss.detach().cpu()

                if loss.detach().cpu() < prev_loss and epoch != 0:
                    state = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': opt.state_dict(),
                        'scheduler' : schedule.state_dict()
                    }
                    save_ckpt(state, True)
                    prev_loss = loss.detach().cpu()

                loss.backward()
                opt.step()
                schedule.step() # in epoch loop
        with Console().status("plotting ...", spinner="aesthetic"):

            data = pd.DataFrame(logs)

            ax=sns.scatterplot(data=data, x="epoch", y="f1", hue="lr")
            # plt.show()
            ax.figure.savefig("lstm_without_embedding/epoch_f1.png", bbox_inches="tight", dpi=500)
            plt.figure()
            ay=sns.scatterplot(data=data, x="epoch", y="loss", hue="lr")
            # plt.show()
            ay.figure.savefig("lstm_without_embedding/epoch_loss.png", bbox_inches="tight", dpi=500)
            az = sns.relplot(data=data, x="epoch", y="lr", kind="line")
            # plt.show()
            plt.savefig("lstm_without_embedding/epoch_lr.png", bbox_inches="tight", dpi=500)


            data["conf_mat"].to_csv("lstm_without_embedding/conf_mat.csv")
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

            # TODO add the model checkpointing capability
#             keras tokens are integers ---> add embedding lstm + pack-pad ...





if __name__=='__main__':
    app.run(main)

