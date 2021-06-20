### Import
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # data visualization
import seaborn as sns  # data visualization
import re
from sklearn.model_selection import train_test_split, KFold, cross_val_score  # tran and test data split
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.svm import SVC  # Support Vector Machine
from sklearn.ensemble import RandomForestClassifier  # Random Rorest Classifier
from sklearn.metrics import roc_auc_score  # ROC and AUC
from sklearn.metrics import accuracy_score  # Accuracy
from sklearn.metrics import recall_score  # Recall
from sklearn.metrics import precision_score  # Prescison
from sklearn.metrics import classification_report  # Classification Score Report
from sklearn.svm import SVC
from rich.console import Console
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.tokenize import word_tokenize
from spacy.lang.en import English
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# Creating our tokenizer function
def spacy_tokenizer(sentence):
    punctuations = string.punctuation

    # Create our list of stopwords
    nlp = spacy.load('en_core_web_sm')
    stop_words = spacy.lang.en.stop_words.STOP_WORDS

    # Load English tokenizer, tagger, parser, NER and word vectors
    parser = English()
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]

    # Removing stop words
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]

    # return a preprocessed list of tokens
    return mytokens


# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()


# Custom transformer using spaCy
# class predictors(TransformerMixin):
#     def transform(self, X, **transform_params):
#         # Cleaning Text
#         return [clean_text(text) for text in X]
#
#     def fit(self, X, y=None, **fit_params):
#         return self
#
#     def get_params(self, deep=True):
#         return {}

class pre_process(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):

        self.columns = columns
    def fit(self, X, y=None, **fitparams):
        return self
    def transform(self, X, **transformparams):

        select_cols = X[self.columns]  # ---> pd.DataFrame
        Console().log(f"shape is ---> {select_cols.shape}")
        # convert to lower case + removal of punctuations
        select_cols = select_cols.apply(lambda x :[" ".join([j.lower().strip().lstrip().rstrip() for j in k.split()]) for k in x])
        # remove the punctuations
        select_cols = select_cols.apply(lambda x: [k.translate(str.maketrans('','',string.punctuation)) for k in x])
        # remove the extra spaces
        select_cols = select_cols.apply(lambda x : [re.sub(' +', ' ', k) for k in x])
        # remove the stopwords
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #                   stop words removal removes many
        #                     important keywords
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # select_cols = select_cols.apply(lambda x : [" ".join([word for word in word_tokenize(k) if not word in STOP_WORDS]) for k in x])
        # removal of frequent words

        return select_cols
m = make_pipeline(pre_process(["Commands"]))


def main():
    with Console().status("pre-processing ....", spinner="bouncingBall"):
        data = pd.read_csv(Path("dataset/validation_data/val.csv").as_posix(), skipinitialspace=True)
        data = data[data["Main Label"] == "textbox"]
        data = data.drop(columns=["Main Label"])
        Console().print(data.shape)
        Console().print(f"columns are ----> {data.columns.tolist()}")
        Console().print(f"columns dtypes ----> {data.dtypes}")

        enc = LabelEncoder()
        # pipeline doesnot operate on the labels
        Y = enc.fit_transform(data['Sub label'].tolist())
        # make column transformer to process the 'commands' column.
        trf_command = make_column_transformer((predictors(), "Commands"))

        process_commands = Pipeline(steps=[("commands_column", predictors()),
                                           ('count', CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 4))),
                                           ('clf', SVC())])

    #
    # with Console().status("training ....", spinner="aesthetic"):
    #     # data = process_commands.fit_transform(data['Commands'].tolist())
    #     folds = 3
    #     cv = KFold(n_splits=(folds - 1))
    #     scores = cross_val_score(process_commands, data['Commands'].tolist(), Y, cv=cv)
    #     # data = process_commands.fit_transform(data['Commands'].tolist())
    #     Console().print(scores)


if __name__ == '__main__':
    main()
