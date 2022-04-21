import json
import subprocess

import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataLoader():
    def __init__(self, file='corpus-webis-tldr-17.json', verbose=False):
        self.file = file
        self.verbose = verbose
        self.get_line_count()

    def get_line_count(self):
        p = subprocess.Popen(['wc', '-l', self.file],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        result, err = p.communicate()
        if p.returncode != 0:
            raise IOError(err)
        self.line_count = int(result.strip().split()[0])

    def load_subreddits(self, subreddits=None, save=False, limit=float('inf')):
        X = []
        y = []
        count = 0

        with open('corpus-webis-tldr-17.json', 'r') as data:
            for i, line in enumerate(data):
                if self.verbose and i % 10000 == 0:
                    print(f'Loading {i}/{self.line_count}...', end='\r')
                sample = json.loads(line)
                if subreddits is not None and sample['subreddit'] not in subreddits:
                    continue
                X.append(sample['content'])
                y.append(sample['subreddit'])
                count += 1
                if count >= limit:
                    break
            print(f'{" " * 50}', end='\r')
            print(f'Loaded {self.line_count}/{self.line_count}')

        le = LabelEncoder()
        y = le.fit_transform(y)

        if subreddits is None:
            subreddits = ['all']

        if save:
            with open(f'corpus-{"-".join(subreddits)}.json', 'w') as f:
                json.dump({'X': X, 'y': y, 'target_names': le.classes_}, f)

        return X, y, le.classes_

    def load_random_sample(self, ratio=1, save=False):
        rng = np.random.default_rng(47)
        index_choices = np.sort(
            rng.choice(np.arange(self.line_count), size=int(self.line_count * ratio),
                       replace=False))

        X = []
        y = []

        with open('corpus-webis-tldr-17.json', 'r') as data:
            j = 0
            for i, line in enumerate(data):
                if index_choices[j] == i:
                    sample = json.loads(line)
                    X.append(sample['content'])
                    y.append(sample['subreddit'])
                    j += 1
                    if self.verbose:
                        print(f'{j}/{len(index_choices)}')
                    if j >= len(index_choices):
                        break

        if save:
            with open(f'corpus-{ratio}.json', 'w') as f:
                json.dump({'X': X, 'y': y}, f)

        return X, y


if __name__ == '__main__':
    dl = DataLoader(verbose=True)
    X, y = dl.load_subreddits(subreddits=['leagueoflegends', 'AdviceAnimals'], save=False)
