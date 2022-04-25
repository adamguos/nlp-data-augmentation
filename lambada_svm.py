# Goal: Run SVM again with lambada augmentation and compare it vs base svm
from DataLoader import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import os
import requests
import gtp_2_simple as gpt2


import nlpaug.augmenter.sentence as nas
import pandas as pd


if __name__ == "__main__":
    dl = DataLoader(verbose=True)
    # X, y, classes = dl.load_subreddits(subreddits=['leagueoflegends', 'AdviceAnimals'])
    # dl.export_for_eda(X, y)
    x, y = dl.import_unaltered_reddit()  # format: number, tab, then all the text

    file_name = "lambada_data/reddit.txt"

    with open(file_name, "w") as f:

        for i in range(len(x)):
            text = x[i]
            if y[i] == 1:
                x[i] = "from lol:\t " + text + "\n"
            else:
                x[i] = "from animal advice:\t" + text + "\n"
            f.write(x[i])

    df = pd.DataFrame({"text": x, "label": y})
    train, test = train_test_split(df, test_size=0.2)

    train.to_csv("lambada_data/train.csv")
    test.to_csv("lambada_data/test.csv")

    # replace words in train csv
    # convert traincsv text into .txt file
    # "0: blah blah"
    # "from adviceAnimals: blah blah"
    # for i, row in enumerate(train["text"]):
    #     train["text"][i] = ""

    # first, get gpt and fine-tune it on our test data
    model_name = "124M"
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)

    sess = gpt2.start_tf_sess()

    gpt2.finetune(sess, file_name, model_name=model_name, steps=1000)

    # use lambada code
    gpt_savepath = "checkpoint/run1/"

    aug = nas.LambadaAug(gpt_savepath + "model", threshold=0.3, batch_size=4)
    aug.augment
    x = dl.preprocess_bow(x)

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    classifier = SVC()
    model = Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])

    scores = cross_validate(
        model,
        x,
        y,
        scoring=["accuracy", "precision", "recall", "f1"],
        verbose=1000,
        n_jobs=-1,
    )
    print(scores)
    breakpoint()
