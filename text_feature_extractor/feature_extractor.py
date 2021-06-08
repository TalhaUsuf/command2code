from transformers import GPT2TokenizerFast, BertTokenizerFast
import argparse
from rich.console import Console
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# adding cmd args
arg = argparse.ArgumentParser()
arg.add_argument('-g', '--gpt2', action="store_true", help="if used gpt2 flag then gpt2 tokenizer to get feature")
arg.add_argument('-t', '--tfidf', action="store_true", help="if flag given then tfidf features will be extracted")
arg.add_argument('-b' '--bert', action="store_true", help="if flag given then bert will used as tokenizer")
arg.add_argument('-k' '--keras', action="store_true", help="if flag given then keras will used as tokenizer")

parser = arg.parse_args()


def main():
    # parse and get tokenizer
    tokenizer = parse()
    x, y_major, y_minor = read_file(pth="dataset/Ultimus Work/Commands_with_labels.csv")

    if parser.tfidf:
        # if tfidf is used then it is an instance of pipeline and should be treated differently
        encoded = tokenizer.fit_transform(x)
        joblib.dump(encoded, "features.pkl")
        joblib.dump(y_major, "main_labels.pkl")
        joblib.dump(y_minor, "sub_labels.pkl")
        Console().print("[cyan]Saved features.pkl, main_labels.pkl and sub_labels.pkl in \'command2code\' dir[/cyan]")

    if parser.keras:
        joblib.dump(encoded, "features.pkl")
        joblib.dump(y_major, "main_labels.pkl")
        joblib.dump(y_minor, "sub_labels.pkl")
        Console().print("[cyan]Saved features.pkl, main_labels.pkl and sub_labels.pkl in \'command2code\' dir[/cyan]")


def read_file(pth: str):
    df = pd.read_csv(pth)

    X = df['Commands'].values
    Y_main = df['Main Label'].values
    Y_sub = df['Sub label'].values

    #     TODO add hsitograms

    return X, Y_main, Y_sub


def parse():
    if parser.gpt2:
        Console().print(f"[red]GPT2 will be used as tokenizer[/red]")
        tok = GPT2TokenizerFast.from_pretrained('gpt2')

        # tok._batch_encode_plus()
    elif parser.tfidf:
        Console().print(f"[red]tfidf will be used as vectorizer[/red]")
        tok = Pipeline([
            ("count", CountVectorizer(ngram_range=(1, 4))),
            ("tfidf", TfidfTransformer())
        ])


    elif parser.bert:
        Console().print(f"[red]BERT will be used as tokenizer[/red]")
        tok = BertTokenizerFast.from_pretrained('bert-base-uncased')

    elif parser.keras:
        Console().print(f"[red]Keras.tokenizer will be used as tokenizer[/red]")
        # TODO remove the # from filters
        tok = tf.keras.preprocessing.text.Tokenizer()

    return tok


if __name__ == '__main__':
    main()
