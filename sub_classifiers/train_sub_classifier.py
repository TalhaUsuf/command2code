### Import
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # data visualization
import seaborn as sns  # data visualization
import re
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
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
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.base import TransformerMixin, BaseEstimator
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.tokenize import word_tokenize
from spacy.lang.en import English
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter

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




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                transformer to pre-process
#                      the text
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class pre_process(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        self.lemmatizer = WordNetLemmatizer()
        self.wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
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
        c = Counter()
        for cmd in select_cols.values.tolist():
            # print(cmd)
            for word in cmd[0].split():
                c[word] += 1

        Console().log(f"10 most common words are ==> {c.most_common(10)}")
        FREQ_WORDS = [k for k,w in c.most_common(1)]
        select_cols = select_cols.apply(lambda x:[" ".join([i for i in k.split() if i not in FREQ_WORDS]) for k in x])

        # remove the rare words
        # removes the unique words from corpus so should not be performed
        Console().log(f"10 least common words are ==> {list(reversed(c.most_common()))[:10] }")


        # stemming
        # Stemming refers to reducing a word to its root form.
        stemmer = PorterStemmer()
        select_cols = select_cols.apply(lambda x : [" ".join([stemmer.stem(j) for j in k.split()]) for k in x])

        # Lemmatization
        # Lemmatization is similar to stemming in reducing inflected words to their word stem but differs in the way that it makes sure the root word (also called as lemma) belongs to the language.

        # select_cols = select_cols.apply(lambda x : [" ".join([token.lemma_ for token in self.parser(k)]) for k in x])

        # in advenced lemmatization ---> words are reduced by considering their parts of speech
        select_cols = select_cols.apply(lambda x : [" ".join([self.lemmatizer.lemmatize(word, self.wordnet_map.get(pos[0], wordnet.VERB)) for word, pos in  nltk.pos_tag(k.split())]) for k in x])


        return select_cols





#
# data = pd.read_csv(Path("dataset/validation_data/val.csv").as_posix(), skipinitialspace=True)
# m = make_pipeline(pre_process(["Commands"]))
#
# Console().print(m.transform(data))
# # m.transform(data)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                      end transformer
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%














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

        # process_commands = Pipeline(steps=[("pre_process", pre_process(["Commands"])),
        #                                    # ('count', CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 4))),
        #                                    ('count', CountVectorizer( ngram_range=(1, 4))),
        #                                    ('clf', SVC())])
        process_commands = make_pipeline( pre_process(['Commands']) , CountVectorizer( ngram_range=(1, 4)) , SVC() )


    # with Console().status("training ....", spinner="aesthetic"):
    #     # data = process_commands.fit_transform(data['Commands'].tolist())
    #     folds = 5
    #     cv = KFold(n_splits=(folds - 1))
    #     Console().log(data.shape)
    #     Console().log(len(Y))
    #     scores = cross_val_score(process_commands, data, np.array(Y), cv=cv)
    #     # data = process_commands.fit_transform(data['Commands'].tolist())
    #     Console().print(scores)
    
        pre_process.fit_transform(data, Y)


if __name__ == '__main__':
    main()
