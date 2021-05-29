import tensorflow as tf
import joblib
from rich.console import Console
from absl import app, flags
from absl.flags import FLAGS
from pathlib import Path

flags.DEFINE_string('cmd', "", 'command to be given')


def main(argv):
    def predict(command_):
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

    f, id = predict(FLAGS.cmd)
    print = Console().print
    print()
    print()
    print()
    print(id)
    print(f)


if __name__ == '__main__':
    app.run(main)
