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
from text_utils import read_file, get_tokenizer

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









if __name__ == '__main__':
    Console().print(parser)
    main()
