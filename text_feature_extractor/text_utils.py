def get_tokenizer():
    if parser.g:
        Console().print(f"[red]GPT2 will be used as tokenizer[/red]")
        tok = GPT2TokenizerFast.from_pretrained('gpt2')

    if parser.t:
        Console().print(f"[red]tfidf will be used as vectorizer[/red]")
        tok = Pipeline([
            ("count", CountVectorizer(ngram_range=(1, 4))),
            ("tfidf", TfidfTransformer())
        ])

    if parser.b:
        Console().print(f"[red]BERT will be used as tokenizer[/red]")
        tok = BertTokenizerFast.from_pretrained('bert-base-uncased')

    if parser.k:
        Console().print(f"[red]Keras.tokenizer will be used as tokenizer[/red]")
        # TODO remove the # from filters
        tok = tf.keras.preprocessing.text.Tokenizer()

    return tok



def read_file(pth: str):
    """
    read the csv file given and extract three fields


    Parameters
    ----------
    pth : string
        path to the csv file

    Returns
    -------
    out : tuple
        X, Y_main, Y_sub all are of type np.ndarray with shapes [N, ] each

    """
    df = pd.read_csv(pth)

    X = df['Commands'].values
    Y_main = df['Main Label'].values
    Y_sub = df['Sub label'].values
    #     TODO add hsitograms
    out = (X, Y_main, Y_sub)

    return out


