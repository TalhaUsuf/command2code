from rich.console import Console
from transformers import GPT2TokenizerFast, BertTokenizerFast
import argparse
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import argparse
from sklearn.model_selection import train_test_split


def get_args():
    # adding cmd args
    arg = argparse.ArgumentParser(description="this script uses feature extraction methods")
    arg.add_argument('-g', action="store_true", help="if used gpt2 flag then gpt2 tokenizer to get feature")
    arg.add_argument('-t', action="store_true", help="if flag given then tfidf features will be extracted")
    arg.add_argument('-b', action="store_true", help="if flag given then bert will used as tokenizer")
    arg.add_argument('-k', action="store_true", help="if flag given then keras will used as tokenizer")
    arg.add_argument('-ng', '--ngrams', default=4, type=int, help="ngrams used for tfidf")
    arg.add_argument('-v', '--validate', action="store_true", help="if flag is given then ;"
                                                                   "validation dataset will also be converted to features inside validation_dataset folder")
    parser = arg.parse_args()

    return parser


def get_tokenizer(parser):
    if parser.g:
        Console().print(f"[red]GPT2 will be used as tokenizer[/red]")
        tok = GPT2TokenizerFast.from_pretrained('gpt2')

    if parser.t:
        # tfidf ensures that sequence lengths are equal
        Console().print(f"[red]tfidf will be used as vectorizer[/red]")
        tok = Pipeline([
            ("count", CountVectorizer(ngram_range=(1, parser.ngrams))),
            ("tfidf", TfidfTransformer())
        ])

    if parser.b:
        # In case of bert , zero padding is automatically added to make sequences equal
        Console().print(f"[red]BERT will be used as tokenizer[/red]")
        tok = BertTokenizerFast.from_pretrained('bert-base-uncased')

    if parser.k:
        # IN case of keras, zero padding has to be done separately
        # to make the sequence lengths equal.
        Console().print(f"[red]Keras.tokenizer will be used as tokenizer[/red]")
        # TODO remove the # from filters
        tok = tf.keras.preprocessing.text.Tokenizer(lower=True)

    return tok


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
