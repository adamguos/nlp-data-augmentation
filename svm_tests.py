import json

from DataLoader import DataLoader

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def single_svm_test(X, y):
    dl = DataLoader()
    X = dl.preprocess_bow(X)

    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    classifier = SVC()
    model = Pipeline([('vectorizer', vectorizer), ('classifier', classifier)])

    scores = cross_validate(model,
                            X,
                            y,
                            scoring=['accuracy', 'precision', 'recall', 'f1'],
                            return_train_score=True,
                            verbose=1000,
                            n_jobs=-1)
    for k in scores:
        if type(scores[k]) == np.ndarray:
            scores[k] = scores[k].tolist()

    return scores


def run_all_tests():
    dl = DataLoader()
    sizes = [50, 100, 500, 1000, 5000]
    results = {}

    for size in sizes:
        results[f'eda_{size}'] = single_svm_test(*dl.import_from_eda(size=size))
        results[f'unaltered_{size}'] = single_svm_test(*dl.import_unaltered_reddit(size=size))

    with open('results.json', 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    run_all_tests()
