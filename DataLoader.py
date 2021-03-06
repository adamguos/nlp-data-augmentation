import json
import re
import subprocess
import os
import nltk

nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataLoader:

    def __init__(self, file="corpus-webis-tldr-17.json", verbose=True):
        self.file = file
        self.verbose = verbose
        self.get_line_count()

    def get_line_count(self):
        p = subprocess.Popen(["wc", "-l", self.file],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        result, err = p.communicate()
        if p.returncode != 0:
            raise IOError(err)
        self.line_count = int(result.strip().split()[0])

    def load_subreddits(self, subreddits=None, save=False, limit=float("inf")):
        X = []
        y = []
        count = 0

        with open("corpus-webis-tldr-17.json", "r") as data:
            for i, line in enumerate(data):
                if self.verbose and i % 10000 == 0:
                    print(f"Loading {i}/{self.line_count}...", end="\r")
                sample = json.loads(line)
                if subreddits is not None and sample["subreddit"] not in subreddits:
                    continue
                content = sample["content"].replace("\n", " ").replace("\r", "")
                X.append(content)
                y.append(sample["subreddit"])
                count += 1
                if count >= limit:
                    break
            if self.verbose:
                print(f'{" " * 50}', end="\r")
                print(f"Loaded {self.line_count}/{self.line_count}")

        le = LabelEncoder()
        y = le.fit_transform(y)

        if subreddits is None:
            subreddits = ["all"]

        if save:
            with open(f'corpus-{"-".join(subreddits)}.json', "w") as f:
                json.dump({"X": X, "y": y, "target_names": le.classes_}, f)

        return X, y

    def preprocess_bow(self, X):
        stopwords = nltk.corpus.stopwords.words("english")
        stemmer = SnowballStemmer("english")
        lemmatizer = WordNetLemmatizer()

        for i, _ in enumerate(X):
            if self.verbose and i % 1000 == 0:
                print(f"Preprocessing {i}/{len(X)}...", end="\r")

            text = re.sub(r"[^\w\s]", "", X[i].lower().strip())
            tokens = text.split()
            # tokens = word_tokenize(X[i])
            tokens = [t for t in tokens if t not in stopwords]

            tokens = [stemmer.stem(token) for token in tokens]
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            X[i] = " ".join(tokens)

        if self.verbose:
            print(f'{" " * 50}', end="\r")
            print(f"Preprocessed {len(X)}/{len(X)}")

        return X

    def export_for_eda(self, X, y, max_samples=float('inf')):
        if max_samples:
            path = f"eda_nlp/data/reddit_{max_samples}.txt"
        else:
            path = "eda_nlp/data/reddit.txt"

        with open(path, "w") as f:
            for i, (text, label) in enumerate(zip(X, y)):
                if i >= max_samples:
                    break
                f.write(f"{label}\t{text}\n")

    def export_for_eda_dir(self, X, y, max_samples):

        directory = f"eda_nlp/data/reddit_{max_samples}"
        df = pd.DataFrame(data={"X": X, "Y": y})
        if (max_samples * 10 > len(df.index)):
            max_samples = int(len(df.index) / 10)
            print(max_samples)

        df = df.sample(n=max_samples * 10, ignore_index=True)

        for i in range(10):
            X = df.loc[(i * max_samples):(i + 1) * max_samples - 1, "X"].values
            y = df.loc[(i * max_samples):(i + 1) * max_samples - 1, "Y"].values

            with open(directory + "/{:d}.txt".format(i), "w") as f:
                for (text, label) in zip(X, y):
                    f.write(f"{label}\t{text}\n")

    def import_unaltered_reddit(self, size=None):
        if size:
            path = f"eda_nlp/data/reddit_{size}.txt"
        else:
            path = "eda_nlp/data/reddit.txt"
        X = []
        y = []
        with open(path, "r") as f:
            for line in f:
                tab = line.index("\t")
                label = line[:tab]
                text = line[(tab + 1):]

                X.append(text)
                y.append(int(label))

        return X, y

    def import_from_mt(self, size):
        path = f"eda_nlp/data/mt_reddit_{size}.txt"
        X = []
        y = []
        with open(path, "r") as f:
            for line in f:
                tab = line.index("\t")
                label = line[:tab]
                text = line[(tab + 1):]

                X.append(text)
                y.append(int(label))

        return X, y

    def import_unaltered_reddit_dir(self, size):
        directory = f"eda_nlp/data/reddit_{size}"
        p = re.compile('[0-9].txt')
        for filename in os.listdir(directory):
            if (p.match(filename)):
                path = os.path.join(directory, filename)
                with open(path, "r") as f:
                    X = []
                    y = []
                    for line in f:
                        tab = line.index("\t")
                        label = line[:tab]
                        text = line[(tab + 1):]
                        X.append(text)
                        y.append(int(label))
                yield X, y

    def import_gpt_label_reddit(self):
        path = "eda_nlp/data/reddit.txt"
        X = []
        y = []
        with open(path, "r") as f:
            for line in f:
                tab = line.index("\t")
                label = line[:tab]
                text = line[(tab + 1):]

                if int(label) == 1:
                    text = "from lol:" + text
                else:
                    text = "from animal advice: " + text
                X.append(text)
                y.append(int(label))

        return X, y

    def import_from_eda(self, size=None):
        if size:
            path = f"eda_nlp/data/eda_reddit_{size}.txt"
        else:
            path = "eda_nlp/data/eda_reddit.txt"
        X = []
        y = []
        with open(path, "r") as f:
            for line in f:
                tab = line.index("\t")
                label = line[:tab]
                text = line[(tab + 1):]

                X.append(text)
                y.append(int(label))

        return X, y

    def import_from_eda_dir(self, size):
        directory = f"eda_nlp/data/reddit_{size}"
        p = re.compile('eda_[0-9].txt')
        for filename in os.listdir(directory):
            if (p.match(filename)):
                path = os.path.join(directory, filename)
                with open(path, "r") as f:
                    X = []
                    y = []
                    for line in f:
                        tab = line.index("\t")
                        label = line[:tab]
                        text = line[(tab + 1):]
                        X.append(text)
                        y.append(int(label))
                yield X, y


if __name__ == '__main__':
    dl = DataLoader()
    X, y = dl.load_subreddits(subreddits=['leagueoflegends', 'AdviceAnimals'])
    dl.export_for_eda_dir(X, y, 50)
    dl.export_for_eda_dir(X, y, 100)
    dl.export_for_eda_dir(X, y, 500)
    dl.export_for_eda_dir(X, y, 1000)
    # dl.export_for_eda_dir(X, y, 5000)
