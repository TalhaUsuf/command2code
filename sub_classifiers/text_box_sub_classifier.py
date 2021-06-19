### Import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # data visualization
from sklearn.model_selection import train_test_split # tran and test data split
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


with Console().status("pre-processing ....", spinner="bouncingBall"):
    data = pd.read_csv(Path("dataset/validation_data/val.csv").as_posix(), skipinitialspace=True)
    Console().print(data.shape)
    Console().print(f"columns are ----> {data.columns.tolist()}")




