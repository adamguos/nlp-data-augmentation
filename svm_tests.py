from DataLoader import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def svm_test(X, y):
    dl = DataLoader()
    X = dl.preprocess_bow(X)

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    classifier = SVC()
    model = Pipeline([('vectorizer', vectorizer), ('classifier', classifier)])

    scores = cross_validate(model,
                            X,
                            y,
                            scoring=['accuracy', 'precision', 'recall', 'f1'],
                            return_train_score=True,
                            verbose=1000,
                            n_jobs=-1)
    return scores


def eda():
    dl = DataLoader()
    X, y = dl.import_from_eda()
    svm_test(X, y)


def unaltered():
    dl = DataLoader()
    X, y = dl.import_unaltered_reddit()
    svm_test(X, y)


if __name__ == '__main__':
    unaltered()
