import os
import json
import torch
import matplotlib.pyplot as plt

import database as db
import neural

with open('config.json', 'r') as infile:
    config = json.load(infile)
    CONFIG_TRANSFORMS = config['transforms']
    FORCE_CPU = config['force_cpu']
    LEARNING_RATE = config['learning_rate']
    EPOCHS = config['epochs']
    BATCH_SIZE = config['batch_size']

training_stocks = db.load_txt('./stock_lists/training_tickers.txt')

df = db.download_history(training_stocks, cache_name='training_stocks', start='2000-01-01', interval='1d')

df = db.apply_config_transforms(df, CONFIG_TRANSFORMS) # Adds more data (technicals, etc)

loader_train, loader_valid = neural.generatetrainingLoaders(df, shuffle=True, batch_size=BATCH_SIZE, date='2018-01-02')

if FORCE_CPU:
    device = torch.device('cpu')
    print('force_cpu set to True, CPU used')
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print('GPU is available')
else:
    device = torch.device('cpu')
    print('GPU is not available, CPU used')

n_input = db.count_columns(df, level=1)
model = neural.LSTM(n_input).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_func = torch.nn.MSELoss()

if os.path.exists('model.pt'):
    state = torch.load('model.pt', map_location=device)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    print('LSTM loaded with:')
    print(model)
else:
    print('LSTM intialized with:')
    print(model)

loss_train = []
loss_valid = []

def train_model(epoch_count):
    for epoch in range(epoch_count):
        loader_iter = iter(loader_train)
        iterations_per_epoch = len(loader_train)
        for iteration in range(iterations_per_epoch):
            data, target = next(loader_iter)
            data = data.float().to(device)
            target = target.float().to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
        
            if iteration%100 == 0:
                model.eval()

                data, target = next(iter(loader_valid))
                data = data.float().to(device)
                target = target.float().to(device)

                holdoutoutput = model(data)
                holdoutloss = loss_func(holdoutoutput, target)

                print(f'Iteration and Epoch: {iteration}/{iterations_per_epoch} {epoch}/{epoch_count}....... ', end='')
                print(f'Training Loss: {loss.item():.4f}', end=' ')
                print(f'Validation Loss: {holdoutloss.item():.4f}')
                loss_train.append(loss.item())
                loss_valid.append(holdoutloss.item())

                model.train()

def save_model():
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'n_input': n_input
    }
    torch.save(state, 'model.pt')

train_model(EPOCHS)
save_model()

plt.yscale('log')
plt.plot(loss_train)
plt.plot(loss_valid)
plt.show()