# Acknowledgements

Wei and Zou 2019, [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text
Classification Tasks](https://arxiv.org/abs/1901.11196)
([GitHub](https://github.com/jasonwei20/eda_nlp))

# Setup

Run `chmod +x download.sh` and `./download.sh`

# Load data

```
from DataLoader import DataLoader
dl = DataLoader()
X, y = dl.load_subreddits(subreddits=['leagueoflegends', 'AdviceAnimals'])
```

# Run EDA

```
dl = DataLoader()
X, y = dl.load_subreddits(subreddits=['leagueoflegends', 'AdviceAnimals'])
dl.export_for_eda(X, y) # exports eda_nlp/data/reddit.txt
```

Then run `python eda_nlp/code/augment.py --input=eda_nlp/data/reddit.txt`. Augmented data is
outputted to `eda_nlp/data/eda_reddit.txt`. Refer to [EDA
repo](https://github.com/jasonwei20/eda_nlp) for details about the file format.

To use the augmented data,

```
X, y = dl.import_from_eda()
```
