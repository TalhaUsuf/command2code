import tensorflow as tf
import joblib
from rich.console import Console
from absl import app, flags
from absl.flags import FLAGS
from pathlib import Path

flags.DEFINE_string("cmd", "", "command to be given by the user")


def predict(command_):
    """
    Takes command from user, tokenizes using `tokenizer.pkl`, predicts using trained model from `models` dir.
    then decodes the predicted label from `label_decoder.pkl` thus printing the actual file code content on screen.


    Parameters
    ----------

    command_ : str
        text command given by the user


    Returns
    -------
    fname : str
        file string name of the file predicted to contain the code content
    fstream : str
         actual content of the file containing the code
    """
    loaded_model = tf.keras.models.load_model('./models/weights.epoch60-val_acc1.00.h5')
    tok = joblib.load('tokenizer.pkl')
    cmd = tok.texts_to_sequences([command_])
    # no need of zero padding
    ydecode = joblib.load("label_decode.pkl")
    ydecode = {a: b for b, a in ydecode.items()}
    pred = loaded_model.predict(cmd)
    pred = tf.squeeze(
        pred, axis=None, name=None
    )
    idx = tf.math.argmax(input=pred)
    idx = idx.numpy().tolist()
    fname = ydecode[idx]

    return [fname, open(fname, 'r').read()]

def main(argv):

    if not FLAGS.cmd:

        f, id = predict(FLAGS.cmd)
        print = Console().print
        print()
        print()
        print()
        print(id)
        print(f)


if __name__ == '__main__':
    Console().rule(title="Prediction Script",align="center", style="red on yellow")
    app.run(main)
