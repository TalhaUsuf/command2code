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
arg = argparse.ArgumentParser(description="this script uses feature extraction methods")
arg.add_argument('-g', action="store_true", help="if used gpt2 flag then gpt2 tokenizer to get feature")
arg.add_argument('-t', action="store_true", help="if flag given then tfidf features will be extracted")
arg.add_argument('-b', action="store_true", help="if flag given then bert will used as tokenizer")
arg.add_argument('-k', action="store_true", help="if flag given then keras will used as tokenizer")
parser = arg.parse_args()


def main():
    # parse and get tokenizer
    tokenizer = get_tokenizer()
    x, y_major, y_minor = read_file(pth="dataset/Ultimus Work/Commands_with_labels.csv")  # all are arrays

    if parser.t:
        # if tfidf is used then it is an instance of pipeline and should be treated differently
        encoded = tokenizer.fit_transform(x)
        joblib.dump(encoded, "tfidf_features.pkl")
        joblib.dump(y_major, "main_labels.pkl")
        joblib.dump(y_minor, "sub_labels.pkl")
        Console().print("[cyan]Saved features.pkl, main_labels.pkl and sub_labels.pkl in \'command2code\' dir[/cyan]")

    if parser.k:
        tokenizer.fit_on_texts(texts=x.tolist())
        encoded = tokenizer.texts_to_sequences(texts=x.tolist())
        joblib.dump(encoded, "kerasTok_features.pkl")
        joblib.dump(y_major, "main_labels.pkl")
        joblib.dump(y_minor, "sub_labels.pkl")
        Console().print(
            "[cyan]Saved kerasTok_features.pkl, main_labels.pkl and sub_labels.pkl in \'command2code\' dir[/cyan]")

    if parser.g:
        encoded = tokenizer(x.tolist(), padding=True, add_special_tokens=False, pad_token=0)
        joblib.dump(encoded, "gpt2_features.pkl")
        joblib.dump(y_major, "main_labels.pkl")
        joblib.dump(y_minor, "sub_labels.pkl")
        Console().print(
            "[cyan]Saved gpt2_features.pkl, main_labels.pkl and sub_labels.pkl in \'command2code\' dir[/cyan]")

    if parser.b:
        encoded = tokenizer(x.tolist(), padding=True, add_special_tokens=False)
        joblib.dump(encoded, "bert_features.pkl")
        joblib.dump(y_major, "main_labels.pkl")
        joblib.dump(y_minor, "sub_labels.pkl")
        Console().print(
            "[cyan]Saved bert_features.pkl, main_labels.pkl and sub_labels.pkl in \'command2code\' dir[/cyan]")




def read_file(pth: str):
    """
    read the csv file given and extract three fields


    Parameters
    ----------
    pth : string
        path to the csv file

    Returns
    -------
    out : tuple
        X, Y_main, Y_sub all are of type np.ndarray with shapes [N, ] each

    """
    df = pd.read_csv(pth)

    X = df['Commands'].values
    Y_main = df['Main Label'].values
    Y_sub = df['Sub label'].values
    #     TODO add hsitograms
    out = (X, Y_main, Y_sub)

    return out


def get_tokenizer():
    if parser.g:
        Console().print(f"[red]GPT2 will be used as tokenizer[/red]")
        tok = GPT2TokenizerFast.from_pretrained('gpt2')

    if parser.t:
        Console().print(f"[red]tfidf will be used as vectorizer[/red]")
        tok = Pipeline([
            ("count", CountVectorizer(ngram_range=(1, 4))),
            ("tfidf", TfidfTransformer())
        ])

    if parser.b:
        Console().print(f"[red]BERT will be used as tokenizer[/red]")
        tok = BertTokenizerFast.from_pretrained('bert-base-uncased')

    if parser.k:
        Console().print(f"[red]Keras.tokenizer will be used as tokenizer[/red]")
        # TODO remove the # from filters
        tok = tf.keras.preprocessing.text.Tokenizer()

    return tok


if __name__ == '__main__':
    Console().print(parser)
    main()
