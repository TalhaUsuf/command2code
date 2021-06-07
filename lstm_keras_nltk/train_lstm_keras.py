import os
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import joblib
import shutil
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import seaborn as sns

lemmat = WordNetLemmatizer()
from rich.console import Console
import tensorflow as tf
from pathlib import Path
from rich.console import Console
import numpy as np
import matplotlib.pyplot as plt

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


print = Console().print
data = pd.read_csv("command2code.csv", skipinitialspace=True)
test = pd.read_csv("test.csv", skipinitialspace=True)
stoplist = stopwords.words('english')
# if stop words need to be removed
# TODO --> stop words also remove words like one, two, three etc
remove_stop_words = lambda x: [[k for k in nltk.word_tokenize(sentence) if k not in stoplist] for sentence in x]
# if stemming is needed
lemmatize = lambda x: [[lemmat.lemmatize(k) for k in nltk.word_tokenize(sentence)] for sentence in x]

xtest, ytest = test['Columns'].values.tolist(), test['TargetFile'].values.tolist()

os.makedirs("models", exist_ok=True)

columns = data.columns.to_list()
X = data['Commands'].values.tolist()
Y = data['TargetFile'].values.tolist()
decode = {k: i for i, k in enumerate(set(Y))}
Y = [decode[k] for k in Y]
Y_enc = tf.keras.utils.to_categorical(Y, num_classes=6)
# for test data
ytest = [decode[k] for k in ytest]
ytest = tf.keras.utils.to_categorical(ytest, num_classes=6)
# =====================================================
X = lemmatize(X)
X = [" ".join(sub) for sub in X]

xtest = lemmatize(xtest)
xtest = [" ".join(sub) for sub in xtest]

tok = Tokenizer(lower=True)
tok.fit_on_texts(X)
tok.document_count

x_seq = tok.texts_to_sequences(X)
x_seq = tf.keras.preprocessing.sequence.pad_sequences(x_seq, padding='post')
# TODO ---> use for sklearn classifiers
# X_enc = tok.texts_to_matrix(X, mode='tfidf')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                     tokenizing the test data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xtest = tok.texts_to_sequences(xtest)
xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, padding='post')

model = tf.keras.models.Sequential(layers=[
    tf.keras.layers.Embedding(len(tok.word_index) + 1, 64, mask_zero=True),
    tf.keras.layers.LSTM(units=128, activation='tanh'),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(6, activation="sigmoid"),
])

#
# tf.keras.utils.plot_model(model, show_shapes=bool, to_file="model.png", dpi=600, show_layer_names=True)
#
# import keras
# keras.utils.plot_model(model, show_shapes=bool, to_file="model.png", dpi=600, show_layer_names=True)
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.90,
    staircase=True)

cback = tf.keras.callbacks.ModelCheckpoint(
    filepath="./models/weights.epoch{epoch:02d}-val_acc{val_acc:.2f}.h5", monitor='val_acc', verbose=0,
    save_best_only=True,
    save_weights_only=False, mode='max', save_freq='epoch',
)
logger = tf.keras.callbacks.CSVLogger(
    filename="./models/stats.csv", separator=',', append=False
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=tf.keras.losses.binary_crossentropy,
              metrics=["acc"])
if any(Path("models").iterdir()):
    shutil.rmtree("models")
Path("models").mkdir(exist_ok=True, parents=True)
hist = model.fit(x_seq, Y_enc, batch_size=8, epochs=200, verbose=1, validation_data=(xtest, ytest),
                 callbacks=[cback, logger])

fig, ax = plt.subplots()
fig.set_size_inches(12, 9)
ax.plot(hist.history['acc'], label="train_acc")
ax.plot(hist.history['val_acc'], label="val_acc")
ax.set_title('Accuracy')
plt.legend()
plt.show()

joblib.dump(tok, 'tokenizer.pkl')
joblib.dump(decode, 'label_decode.pkl')
