import json
from statistics import mean, stdev

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


def plot_accuracy():
    with open('results.json', 'r') as f:
        results = json.load(f)

    sizes = [50, 100, 500, 1000, 5000]
    eda_acc = [mean(results[f'eda_{size}']['test_accuracy']) for size in sizes]
    eda_std = [stdev(results[f'eda_{size}']['test_accuracy']) for size in sizes]
    unaltered_acc = [mean(results[f'unaltered_{size}']['test_accuracy']) for size in sizes]
    unaltered_std = [stdev(results[f'unaltered_{size}']['test_accuracy']) for size in sizes]

    x = np.arange(len(eda_acc))
    plt.plot(x, eda_acc)
    plt.plot(x, unaltered_acc)
    plt.legend(['eda', 'unaltered'], loc=4)
    plt.title('Accuracy of EDA and unaltered datasets using SVM')
    plt.xlabel('Dataset size')
    plt.ylabel('Accuracy')
    plt.xticks(x, sizes)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.show()

    print(eda_acc)
    print(unaltered_acc)


if __name__ == '__main__':
    plot_accuracy()
