from transformers import GPT2TokenizerFast
import argparse
from rich.console import Console


arg = argparse.ArgumentParser()
arg.add_argument('-g', '--gpt2', action="store_true", help="if used gpt2 flag then gpt2 tokenizer to get feature")
arg.add_argument('-t', '--tfidf', action="store_true", help="if flag given then tfidf features will be extracted")
arg.add_argument('-b' '--bert', action="store_true", help="if flag given then bert will used as tokenizer")
arg.add_argument('-k' '--keras', action="store_true", help="if flag given then keras will used as tokenizer")

parser = arg.parse_args()


if parser.gpt2:
    Console().print(f"[red]GPT2 will be used as tokenizer[/red]")
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    # tok._batch_encode_plus()
elif parser.tfidf:
    pass

elif parser.bert:
    pass

elif parser.keras:
    pass
