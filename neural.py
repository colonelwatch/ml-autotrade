import json
import multiprocessing as mp
import numpy as np
import tables
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

class historyDataset(torch.utils.data.Dataset):
    def __init__(self, df, path, id : str):
        self.len = len(df.index)-TOTAL_INTERVAL_SIZE+1
        if self.len <= 0:
            self.len = 0 # If a dataset is invalid, this prevents an error and omits the dataset
            return
        
        self.path = path
        self.id = id
        self.example_shape = (TOTAL_INTERVAL_SIZE, len(df.columns))
        self.close_index = list(df.columns).index('Close')
        
        params = {
            'example_shape': self.example_shape,
            'num_samples': self.len
        }
        h5file = tables.open_file(self.path, 'a')
        try:
            # Raises tables.exceptions.NodeError if already exists
            atom = tables.Atom.from_dtype(np.dtype((np.float32, self.example_shape)))
            h5arr = h5file.create_carray('/', self.id, atom, (self.len,))

            arr = df.values
            batch_arr = np.array([arr[i:i+TOTAL_INTERVAL_SIZE] for i in range(self.len)])
            with np.errstate(divide='ignore', invalid='ignore'):
                batch_norm_arr = db.batch_apply(batch_arr, db.masked_normalize)
                batch_norm_arr = np.nan_to_num(batch_norm_arr, copy=False, nan=0, posinf=0, neginf=0)
            h5arr[:] = batch_norm_arr

            h5file.set_node_attr('/'+self.id, 'params', params)
        except tables.exceptions.NodeError: # In case the node already exists, reuse it...
            # but throw an error if the cache was made with different settings
            if h5file.get_node_attr('/'+self.id, 'params') != params:
                raise ValueError(f'{self.id} cache params do not match current settings')
        h5file.close()
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        h5file = tables.open_file(self.path, 'r')

        arr_idx = h5file.get_node('/', self.id)[idx]
        input_data = arr_idx[0:INTERVAL_TENSOR_SIZE]
        label = arr_idx[INTERVAL_TENSOR_SIZE][self.close_index]

        h5file.close()
        return input_data, label

def generate_split_loaders(df, shuffle, batch_size, **kwargs):
    datasets_training = []
    datasets_validation = []
    
    # Generates a training and validation dataset for each ticker
    for ticker, df_t in df.groupby(axis=1, level=0):
        df_t = df_t.dropna()
        df_t = df_t.droplevel(level=0, axis=1) # Removes ticker level
        
        df_t_training, df_t_validation = db.split(df_t, **kwargs)

        datasets_training.append(historyDataset(df_t_training, './cache/historyDataset_train_cache.h5', ticker))
        datasets_validation.append(historyDataset(df_t_validation, './cache/historyDataset_valid_cache.h5', ticker))
    
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