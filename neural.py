import json
import torch

import database as db

# Constant declarations to configure the model, other files access these

with open('config.json', 'r') as infile:
    config = json.load(infile)
    
    INTERVAL_TENSOR_SIZE = config['model']['input_timesteps']
    TOTAL_INTERVAL_SIZE = 1+INTERVAL_TENSOR_SIZE
    HIDDEN_SIZE = config['model']['hidden_size']
    LAYER_COUNT = config['model']['layer_count']
    NUM_WORKERS = config['num_workers']

# Model definition

class LSTM(torch.nn.Module):
    def __init__(self, n_input):
        super(LSTM, self).__init__()
        
        self.n_input = n_input
        self.n_hidden = HIDDEN_SIZE
        self.n_layers = LAYER_COUNT

        self.lstm = torch.nn.LSTM(
            input_size = self.n_input,
            hidden_size = self.n_hidden,
            num_layers = self.n_layers,
            batch_first = True
        )

        self.linear = torch.nn.Linear(self.n_hidden, 1)
    def forward(self, input_tensor):
        out, (h_n, h_c) = self.lstm(input_tensor, None)

        out = self.linear(out)[:, -1, :].view(-1)

        return out

# Dataset class definition and Dataloader generator functions

# Note: The convention is to normalize from the start, but this method normalizes on the fly.
#  This makes it very slow, but this probably keeps future information from leaking in.
class historyDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.len = len(df.index)-TOTAL_INTERVAL_SIZE+1
        if self.len <= 0:
            self.len = 0 # If a dataset is invalid, this prevents an error and omits the dataset
            return
        self.df = df
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        df = db.sub_df(self.df, idx, TOTAL_INTERVAL_SIZE)
        # df = df/df.iloc[0]-1
        df = db.normalize(df, exclude_label=True)

        input_data = df.iloc[0:INTERVAL_TENSOR_SIZE].values
        label = df['Close'][INTERVAL_TENSOR_SIZE]
        
        return input_data, label

def generate_split_loaders(df, shuffle, batch_size, **kwargs):
    datasets_training = []
    datasets_validation = []
    
    # Generates a training and validation dataset for each ticker
    for foo, df_t in df.groupby(axis=1, level=0):
        df_t = df_t.dropna()
        df_t = df_t.droplevel(level=0, axis=1) # Removes ticker level
        
        df_t_training, df_t_validation = db.split(df_t, **kwargs)

        datasets_training.append(historyDataset(df_t_training))
        datasets_validation.append(historyDataset(df_t_validation))
    
    dataset_training = torch.utils.data.ConcatDataset(datasets_training)
    loader_training = torch.utils.data.DataLoader(dataset_training, batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        drop_last=True)
    dataset_validation = torch.utils.data.ConcatDataset(datasets_validation)
    loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        drop_last=True)

    return loader_training, loader_validation