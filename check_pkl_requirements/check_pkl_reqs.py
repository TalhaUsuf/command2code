"""
This script checks if all requried pickle files exist inside the command2code dir.
"""
from pathlib import Path
import os


def check_requirements():
    list_pkl_files = list(Path(".").glob("*.pkl"))
    FILES = ['keras_tokenizer.pkl', 'enc_sub.pkl', 'sub_labels.pkl', 'bert_tokenizer.pkl', 'enc_main.pkl',
             'main_labels.pkl', 'tfidf_tokenizer.pkl', 'kerasTok_features.pkl', 'tfidf_features.pkl',
             'bert_features.pkl']

    count_existing_files = [str(k) for k in list_pkl_files if str(k) in FILES]

    assert len(count_existing_files) == len(
        FILES), f"No. of *.pkl files not equal \n following files should exist: \n {FILES}"

    print("OK")


if __name__ == '__main__':
    check_requirements()
