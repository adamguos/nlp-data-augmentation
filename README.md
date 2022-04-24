# Setup

Run `chmod +x download.sh` and `./download.sh`

# Load data

```
from DataLoader import DataLoader
dl = DataLoader()
X, y = dl.load_subreddits(subreddits=['leagueoflegends', 'AdviceAnimals', ...], save=False)
```
