import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tqdm import trange


def load_data():

    dataset = pd.read_csv("../project1/commands2code.csv", skipinitialspace=True)

    commands = dataset['Commands'].values.tolist()
    labels = dataset['TargetFile'].values.tolist()


    tok = tf.keras.preprocessing.text.Tokenizer(lower=True)
    tok.fit_on_texts(texts=commands)
    commands_tok = tok.texts_to_sequences(texts=commands)
    seq_len = [len(k) for k in commands_tok]
    vocab = len(tok.word_index) + 1
    X = tf.keras.preprocessing.sequence.pad_sequences(commands_tok, padding='post', truncating='post')
    enc = LabelEncoder()
    Y = enc.fit_transform(labels)
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)
    bs = X.shape[0]

    return X,Y, weights, seq_len, vocab, bs


