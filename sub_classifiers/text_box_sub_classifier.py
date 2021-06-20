### Import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # data visualization
from sklearn.model_selection import train_test_split, KFold, cross_val_score # tran and test data split
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.svm import SVC #Support Vector Machine
from sklearn.ensemble import RandomForestClassifier # Random Rorest Classifier
from sklearn.metrics import roc_auc_score # ROC and AUC
from sklearn.metrics import accuracy_score # Accuracy
from sklearn.metrics import recall_score # Recall
from sklearn.metrics import precision_score # Prescison
from sklearn.metrics import classification_report # Classification Score Report
from sklearn.svm import SVC
from rich.console import Console
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
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
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return a preprocessed list of tokens
    return mytokens

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}



with Console().status("pre-processing ....", spinner="bouncingBall"):
    data = pd.read_csv(Path("dataset/validation_data/val.csv").as_posix(), skipinitialspace=True)
    data = data[data["Main Label"]=="textbox"]
    data = data.drop(columns=["Main Label"])
    Console().print(data.shape)
    Console().print(f"columns are ----> {data.columns.tolist()}")
    Console().print(f"columns dtypes ----> {data.dtypes}")

    enc = LabelEncoder()
    Y = enc.fit_transform(data['Sub label'].tolist())
    process_commands = Pipeline(steps=[("commands_column", predictors()),
                                       ('count', CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 4))),
                                       ('clf', SVC())])

    # data = process_commands.fit_transform(data['Commands'].tolist())
    folds = 5
    cv = KFold(n_splits=(folds - 1))
    scores = cross_val_score(process_commands, data['Commands'].tolist(), Y, cv=cv)
    # data = process_commands.fit_transform(data['Commands'].tolist())
    Console().print(scores)
