from DataLoader import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

if __name__ == '__main__':
    dl = DataLoader(verbose=True)
    X, y, classes = dl.load_subreddits(subreddits=['leagueoflegends', 'AdviceAnimals'], save=False)
    dl.export_for_eda(X, y)
    X = dl.preprocess_bow(X)

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    classifier = SVC()
    model = Pipeline([('vectorizer', vectorizer), ('classifier', classifier)])

    scores = cross_validate(model, X, y, scoring=['accuracy', 'precision', 'recall', 'f1'])
    print(scores)
    breakpoint()
