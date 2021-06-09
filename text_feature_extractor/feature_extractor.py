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
from text_utils import read_file, get_tokenizer, get_args
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def main():
    """
    gets parsers according to the flag given by user and saves the tokenizers and label encoders accordingly

    Tokenizers Supported
    --------------------
    tfidf_tokenizer
        uses sklearn pipeline
    keras_tokenizer
        uses tf.keras processing functions. The features are 0 padded to max-len in batch
    gpt2_tokenizer
        uses transformers library, currently dysfunctional
    bert_tokenizer
        uses transformers library, saves features with dictionary having following keys : 'input_ids'\
        'token_type_ids' and 'attention_mask'

    Returns
    -------
    None :
        Saves the pickle files inside ``command2code`` dir.
    """
    # parse and get tokenizer
    parser = get_args()
    tokenizer = get_tokenizer(parser)
    x, y_major, y_minor = read_file(pth="dataset/Ultimus Work/Commands_with_labels.csv")  # all are arrays

    # 1) encode the labels
    encoder_main = LabelEncoder()
    encoder_sub = LabelEncoder()

    y_major_enc =  encoder_main.fit_transform(y_major)
    y_minor_enc =  encoder_sub.fit_transform(y_minor)

    joblib.dump(encoder_main, "enc_main.pkl")
    joblib.dump(encoder_sub, "enc_sub.pkl")
    Console().print(f"Saved [cyan]enc_main.pkl[/cyan] and [magenta]enc_sub.pkl[/magenta]")
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                      fit on the whole corpus because
    #               test set will be given separately
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if parser.t:
        # if tfidf is used then it is an instance of pipeline and should be treated differently
        encoded = tokenizer.fit_transform(x)
        joblib.dump(tokenizer, "tfidf_tokenizer.pkl")
        joblib.dump(encoded, "tfidf_features.pkl")
        joblib.dump(y_major, "main_labels.pkl")
        joblib.dump(y_minor, "sub_labels.pkl")
        Console().print("[cyan]Saved tfidf_features.pkl, main_labels.pkl and sub_labels.pkl in \'command2code\' dir[/cyan]")

    if parser.k:
        # for keras tokenizer do zero-padding
        tokenizer.fit_on_texts(texts=x.tolist())
        encoded = tokenizer.texts_to_sequences(texts=x.tolist())
        # have to do zero padding manually
        encoded = tf.keras.preprocessing.sequence.pad_sequences(encoded, padding="post")
        joblib.dump(tokenizer, "keras_tokenizer.pkl")
        joblib.dump(encoded, "kerasTok_features.pkl")
        joblib.dump(y_major, "main_labels.pkl")
        joblib.dump(y_minor, "sub_labels.pkl")
        Console().print(
            "[cyan]Saved kerasTok_features.pkl, main_labels.pkl and sub_labels.pkl in \'command2code\' dir[/cyan]")

    if parser.g:
        encoded = tokenizer(x.tolist(), padding=True, add_special_tokens=False, pad_token=0)
        joblib.dump(tokenizer, "gpt2_tokenizer.pkl")
        joblib.dump(encoded, "gpt2_features.pkl")
        joblib.dump(y_major, "main_labels.pkl")
        joblib.dump(y_minor, "sub_labels.pkl")
        Console().print(
            "[cyan]Saved gpt2_features.pkl, main_labels.pkl and sub_labels.pkl in \'command2code\' dir[/cyan]")

    if parser.b:

        encoded = tokenizer(x.tolist(), padding=True, add_special_tokens=False)
        joblib.dump(tokenizer, "bert_tokenizer.pkl")
        joblib.dump(encoded, "bert_features.pkl")
        joblib.dump(y_major, "main_labels.pkl")
        joblib.dump(y_minor, "sub_labels.pkl")
        Console().print(
            "[cyan]Saved bert_features.pkl, main_labels.pkl and sub_labels.pkl in \'command2code\' dir[/cyan]")









if __name__ == '__main__':

    main()
