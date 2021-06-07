from transformers import GPT2TokenizerFast
import argparse


arg = argparse.ArgumentParser()
arg.add_argument('-g','--gpt2', action="store_true", help="if use gpt2 tokenizer to get feature")
parser = arg.parse_args()


if parser.gpt2:
    print(f"USE GPT2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    print([k for k in dir(tokenizer) if not k.startswith("_")])

