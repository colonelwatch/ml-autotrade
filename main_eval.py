import json
import torch
import numpy as np
from matplotlib import pyplot as plt

import database as db
import neural

def main():
    with open('config.json', 'r') as infile:
        config = json.load(infile)
        CONFIG_TRANSFORMS = config['transforms']

    eval_stocks = db.load_txt('./stock_lists/training_tickers.txt')

    df = db.download_history(eval_stocks, cache_name='eval_stocks', start='2018-01-02', interval='1d')

    df = db.apply_config_transforms(df, CONFIG_TRANSFORMS) # Adds more data (technicals, etc)

    # Produces datasets each of the histories then samples randomly from them
    datasets = []
    for ticker in df.columns.levels[0]:
        datasets.append(neural.historyDataset(df[ticker].dropna(), './cache/historyDataset_eval_cache.h5' , ticker))
    dataset = torch.utils.data.ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True, drop_last=True)

    state = torch.load('model.pt')
    n_input = state['n_input']
    model = neural.LSTM(n_input) # Resets model with a batch size of target list length
    model.load_state_dict(state['model'])
    model.eval()
    print('LSTM loaded')

    data, label = next(iter(loader))
    data = data.float()

    out_norm = model(data).detach().view(-1).numpy()

    last = data[:, -1, 0].numpy()
    label = label.view(-1).numpy()

    dx_label = label-last
    dx_norm = out_norm-last

    accuracy_norm = (np.sum(np.logical_and(dx_label < 0, dx_norm < 0))+np.sum(np.logical_and(dx_label > 0, dx_norm > 0))) / 100
    print(f'Accuracy of model: {accuracy_norm:.3f}')

    accuracy_alwaysup = np.sum(np.sum(np.logical_and(dx_label > 0, 1)))/100
    print(f'Accuracy of always predicting up: {accuracy_alwaysup:.3f}')

    rng = np.random.default_rng()
    random = rng.integers(2, size=100)
    accuracy_random = (np.sum(np.logical_and(dx_label < 0, random))+np.sum(np.logical_and(dx_label > 0, random))) / 100
    print(f'Accuracy of always predicting randomly: {accuracy_random:.3f}')

    # Below are two ways to order the data, the correlation in both representations should be strong

    # Ordered by predicted change
    p = np.argsort(dx_label)
    dx_label = dx_label[p]
    dx_norm = dx_norm[p]

    plt.plot(dx_label, '.r')
    plt.plot(dx_norm, '.g')
    plt.plot(np.zeros(100))
    plt.show()

    # Ordered by true change
    p = np.argsort(dx_norm)
    dx_label = dx_label[p]
    dx_norm = dx_norm[p]

    plt.plot(dx_label, '.r')
    plt.plot(dx_norm, '.g')
    plt.plot(np.zeros(100))
    plt.show()

if __name__ == '__main__':
    main()