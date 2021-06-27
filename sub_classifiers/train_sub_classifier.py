### Import
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # data visualization
import seaborn as sns  # data visualization
import re
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import nltk
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import joblib
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve  # tran and test data split
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.svm import SVC  # Support Vector Machine
from sklearn.ensemble import RandomForestClassifier  # Random Rorest Classifier
from sklearn.metrics import roc_curve, confusion_matrix  # ROC and AUC
from sklearn.metrics import accuracy_score, precision_recall_curve  # Accuracy
from sklearn.metrics import recall_score  # Recall
from sklearn.metrics import precision_score  # Prescisonfr
from sklearn.preprocessing import label_binarize
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
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.tokenize import word_tokenize
from spacy.lang.en import English
import string
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


# # Creating our tokenizer function
# def spacy_tokenizer(sentence):
#     punctuations = string.punctuation
# 
#     # Create our list of stopwords
#     nlp = spacy.load('en_core_web_sm')
#     stop_words = spacy.lang.en.stop_words.STOP_WORDS
# 
#     # Load English tokenizer, tagger, parser, NER and word vectors
#     parser = English()
#     # Creating our token object, which is used to create documents with linguistic annotations.
#     mytokens = parser(sentence)
# 
#     # Lemmatizing each token and converting each token into lowercase
#     mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
# 
#     # Removing stop words
#     mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]
# 
#     # return a preprocessed list of tokens
#     return mytokens


# Basic function to clean the text
# def clean_text(text):
#     # Removing spaces and converting text into lowercase
#     return text.strip().lower()


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

    def transform(self, X):

        select_cols = X[self.columns]  # ---> pd.DataFrame
        Console().log(f"shape is ---> {select_cols.shape}")
        # convert to lower case + removal of punctuations
        select_cols = select_cols.apply(
            lambda x: [" ".join([j.lower().strip().lstrip().rstrip() for j in k.split()]) for k in x])
        # remove the punctuations
        select_cols = select_cols.apply(lambda x: [k.translate(str.maketrans('', '', string.punctuation)) for k in x])
        # remove the extra spaces
        select_cols = select_cols.apply(lambda x: [re.sub(' +', ' ', k) for k in x])
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
        FREQ_WORDS = [k for k, w in c.most_common(1)]
        select_cols = select_cols.apply(lambda x: [" ".join([i for i in k.split() if i not in FREQ_WORDS]) for k in x])

        # remove the rare words
        # removes the unique words from corpus so should not be performed
        Console().log(f"10 least common words are ==> {list(reversed(c.most_common()))[:10]}")

        # stemming
        # Stemming refers to reducing a word to its root form.
        stemmer = PorterStemmer()
        select_cols = select_cols.apply(lambda x: [" ".join([stemmer.stem(j) for j in k.split()]) for k in x])

        # Lemmatization
        # Lemmatization is similar to stemming in reducing inflected words to their word stem but differs in the way that it makes sure the root word (also called as lemma) belongs to the language.

        # select_cols = select_cols.apply(lambda x : [" ".join([token.lemma_ for token in self.parser(k)]) for k in x])

        # in advenced lemmatization ---> words are reduced by considering their parts of speech
        select_cols = select_cols.apply(lambda x: [" ".join(
            [self.lemmatizer.lemmatize(word, self.wordnet_map.get(pos[0], wordnet.VERB)) for word, pos in
             nltk.pos_tag(k.split())]) for k in x])

        return select_cols


class Reshape(BaseEstimator, TransformerMixin):
    def __init__(self):
        super(Reshape, self).__init__()

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        ''' because count vectorizer needs in this form'''
        # X ---> dataframe
        flatten = lambda x: [k for k in x.values.squeeze().tolist()]

        out = flatten(X)
        return out


# class Vectorizer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         super(Vectorizer, self).__init__()
#
#     def fit(self, X, y=None):
#         self.vec = CountVectorizer(ngram_range=(1, 4))
#         return self
#
#     def transform(self, X):
#         return self.vec.fit_transform(X).toarray()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.2, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="neg_mean_squared_error")
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = -np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = -np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


class naive_bayesian(BaseEstimator, ClassifierMixin):
    def __init__(self, clf):
        super(naive_bayesian, self).__init__()
        self.clf = clf

    def fit(self, X, y=None):
        self.classifier_ = self.clf
        self.classifier_.fit(X.A, y)
        return self

    def predict(self, X):
        # return self.classifier_.predict_proba(X)
        return self.classifier_.predict(X)
    def predict_proba(self, X):
        return self.classifier_.predict_proba(X.A)
        # return self.classifier_.predict(X)


def main():
    def plot_confusion_matrix(cm,
                              target_names,

                              title='Confusion matrix',
                              cmap=None,
                              normalize=True):
        """
        given a sklearn confusion matrix (cm), make a nice plot

        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix

        target_names: given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                      see http://matplotlib.org/examples/color/colormaps_reference.html
                      plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                      If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                  # sklearn.metrics.confusion_matrix
                              normalize    = True,                # show proportions
                              target_names = y_labels_vals,       # list of names of the classes
                              title        = best_estimator_name) # title of graph

        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        """
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools

        accuracy = np.trace(cm) / np.sum(cm).astype('float')
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.savefig(f"./confusion_matrix_{tag}.png", dpi=400, bbox_inches="tight")
        plt.show()

    with Console().status("pre-processing ....", spinner="bouncingBall"):
        original_data = pd.read_csv(
            Path("/home/talha/PycharmProjects/command2code/dataset/validation_data/command2code.csv").as_posix(),
            skipinitialspace=True)
        tags = original_data["Main Label"].unique().tolist()

        for tag in tags:

            # tag = "radiogroup"
            data = original_data.copy()
            data = data[data["Main Label"] == tag]
            data = data.drop(columns=["Main Label"])
            Console().print(data.shape)
            Console().print(f"columns are ----> {data.columns.tolist()}")
            Console().print(f"columns dtypes ----> {data.dtypes}")

            enc = LabelEncoder()
            # pipeline doesnot operate on the labels

            # Y = label_binarize(data['Sub label'].tolist(), classes=[*range(3)])
            Y = enc.fit_transform(data['Sub label'].tolist())

            # make column transformer to process the 'commands' column.
            vec = CountVectorizer(ngram_range=(1, 4))
            clf = SVC()

            svc = Pipeline([
                ("vec", CountVectorizer(ngram_range=(1, 4))),
                ("tfidf", TfidfTransformer()),
                ("svc", SVC(probability=True))
            ])
            logistic = Pipeline([
                # ("vec", Vectorizer()),
                ("vec", CountVectorizer(ngram_range=(1, 4))),
                ("tfidf", TfidfTransformer()),
                ("logistic_CV", LogisticRegressionCV(cv=5, multi_class="ovr"))
            ])

            NB = Pipeline([
                # ("vec", Vectorizer()),
                ("vec", CountVectorizer(ngram_range=(1, 4))),
                ("tfidf", TfidfTransformer()),
                ("NB", naive_bayesian(GaussianNB()))
            ])

            process_commands = Pipeline(steps=[("pre_process", pre_process(["Commands"])),
                                               ("reshape", Reshape()),
                                               ])

            X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.30, stratify=Y)

            processed_X_train = process_commands.fit_transform(X_train, y_train)
            processed_X_test = process_commands.fit_transform(X_test, y_test)

            # NB.fit(processed_X_train, y_train)

            votingC = VotingClassifier(estimators=[('svc', svc), ('logistic', logistic),
                                                   ('nb', NB),
                                                   ],
                                       voting='soft', n_jobs=4, )
            #
            votingC.fit(processed_X_train, y_train)
            y_pred = votingC.predict(processed_X_test)


            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #                      confusion matrix
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            cf = confusion_matrix(y_test, y_pred)

            plot_confusion_matrix(cf,
                                  target_names=enc.classes_,
                                  title='Confusion matrix',
                                  cmap=None,
                                  normalize=True)

            Console().print(y_pred)
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #                      classification report
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            print(classification_report(y_test, y_pred, target_names=enc.classes_))
            probs = votingC.predict_proba(processed_X_test)
            # Console().print(probs)

            sns.set_theme()

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #                      precision-recall curve
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            plt.figure(0)
            # precision recall curve
            n_classes = probs.shape[-1]
            dummy = np.eye(n_classes)[y_test]
            precision = dict()
            recall = dict()
            for i in range(n_classes):
                precision[i], recall[i], _ = precision_recall_curve(dummy[:, i],
                                                                    probs[:, i])
                plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

            plt.xlabel("recall")
            plt.ylabel("precision")
            plt.legend(loc="best")
            plt.title("precision vs. recall curve")
            plt.savefig(f"./precision_recall_{tag}.png", bbox_inches="tight", dpi=400)
            plt.show()
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #                      ROC curve
            #                      calibration curve
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            plt.figure(1)
            n_classes = probs.shape[-1]
            dummy = np.eye(n_classes)[y_test]
            tpr = dict()
            fpr = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(dummy[:, i],
                                              probs[:, i])
                plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))

            plt.xlabel("fpr")
            plt.ylabel("tpr")
            plt.legend(loc="best")
            plt.title("ROC curve")
            plt.savefig(f"./roc_curve_{tag}.png", bbox_inches="tight", dpi=400)
            plt.show()

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #                      calibration curve
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            plt.figure(2, figsize=(10, 10))
            ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=3)
            ax2 = plt.subplot2grid((3, 3), (2, 0), colspan=3)

            fraction_of_positives = dict()
            mean_predicted_value = dict()
            for i in range(n_classes):
                fraction_of_positives[i], mean_predicted_value[i] = calibration_curve(dummy[:, i], probs[:, i], n_bins=10)
                ax1.plot(mean_predicted_value[i], fraction_of_positives[i], "s-",
                         label=f"{enc.classes_[i]}")
                ax1.scatter(np.linspace(0, 1, 50), np.linspace(0, 1, 50), marker="p", s=80, facecolor='green', alpha=0.5,
                            edgecolor="k")
                ax2.hist(probs[:, i], range=(0, 1), bins=10, label=f"{enc.classes_[i]}",
                         histtype="step", lw=2)

            ax1.set_ylabel("Fraction of positives")
            ax1.set_ylim([-0.05, 1.05])
            ax1.legend(loc="lower right")
            ax1.set_title('Calibration plots (reliability curve)')

            ax2.set_xlabel("Mean predicted value")
            ax2.set_ylabel("Count")
            ax2.legend(loc="upper center", ncol=2)

            plt.tight_layout()
            plt.savefig("./calibration_curve.png", bbox_inches="tight", dpi=400)
            plt.show()

            joblib.dump(enc, f"./sub_classifiers/label_encoder_{tag}.pkl")
            joblib.dump(votingC, f"./sub_classifiers/sub_clf_{tag}.pkl")
            joblib.dump(process_commands, f"./sub_classifiers/process_commands_{tag}.pkl")

if __name__ == '__main__':
    main()
