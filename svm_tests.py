import os
from DataLoader import DataLoader
import numpy as np
import pandas as pd
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

    return scores['test_accuracy'].mean()

def run_svm_tests():
    dl = DataLoader()
    sizes = [50, 100, 500, 1000, 5000]
    file_name = 'svm_scores.csv'

    da_methods = {'eda': dl.import_from_eda, 'unaltered': dl.import_unaltered_reddit}

    if os.path.exists(file_name):
        df = pd.read_csv(file_name, index_col=0)
    else:
        df = pd.DataFrame(columns=da_methods.keys())

    for size in sizes:
        if size not in df.index:
            df.loc[size] = np.nan

        for method_name in da_methods:
            da_method = da_methods[method_name]

            if method_name not in df.columns:
                df.insert(loc=0, column=method_name, value=np.nan)

            if np.isnan(df.loc[size][method_name]):
                X, y = da_method(size=size)
                df.loc[size][method_name] = single_svm_test(X, y)

            print(df)
            df.to_csv(file_name)

def run_svm_tests_dir():
    dl = DataLoader()
    sizes = [50,100,500,1000]
    file_name = 'svm_scores_many.csv'
    da_methods = {'eda': dl.import_from_eda_folder, 'unaltered': dl.import_unaltered_reddit_folder}

    dat = []
    for size in sizes: 
        row = [] 
        for method_name in da_methods:
            da_method = da_methods[method_name]
            col = []
            for X,y in da_method(size=size):
                col.append(single_svm_test(X, y))
            row.append(col)
        dat.append(row)
    
    df = pd.DataFrame(dat,columns = ["eda_means","unaltered_means"])
    df.index = sizes
    df.to_csv(file_name)
    return df

if __name__ == '__main__':
    run_svm_tests()
