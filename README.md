# Acknowledgements

[Wei and Zou 2019, EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/abs/1901.11196)

[GitHub](https://github.com/jasonwei20/eda_nlp)

# Setup

Run `chmod +x download.sh` and `./download.sh`

# Load data

```
from DataLoader import DataLoader
dl = DataLoader()
X, y = dl.load_subreddits(subreddits=['leagueoflegends', 'AdviceAnimals', ...], save=False)
```
