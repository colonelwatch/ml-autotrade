import os
import datetime
import json
import numpy as np
import yfinance as yf
import pandas as pd # hidden requirement: tables

import technicals as tech
import moredata

def load_txt(filename):
    loaded_list = open(filename).read().split('\n')
    return [item for item in loaded_list if item] # Removes empty strings 

def store_h5(filename, df, **kwargs):
    df.to_hdf(filename, 'history', 'w')
    with open(filename.replace('.h5', '_metadata.json'), 'w') as outfile:
        json.dump(dict(**kwargs), outfile)

def load_h5(filename):
    data = pd.read_hdf(filename)
    with open(filename.replace('.h5', '_metadata.json'), 'r') as infile:
        metadata = json.load(infile)
    return data, metadata

# <MultiIndex Functions>

# If a cache_name is defined, then a cache of that name will be created or used instead of downloading
def download_history(tickers, cache_name=None, enforce_start=False, **kwargs):
    if cache_name:
        kwargs['tickers'] = tickers # Integrates tickers into kwargs for comparison

        dt = datetime.date.today()
        filename = f'./cache/{dt}_{cache_name}.h5'
        try:
            df, kwargs_loaded = load_h5(filename)
            if kwargs_loaded == kwargs:
                print('Cached data loaded')
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            print('Cached data not found, retrieving data with yfinance...')
            df = yf.download(**kwargs, group_by='ticker')

            if not os.path.exists('./cache/'):
                os.mkdir('./cache/')
            store_h5(filename, df, **kwargs) # Includes tickers with kwargs
    else:
        print('Retrieving data...')
        df = yf.download(tickers=tickers, **kwargs, group_by='ticker')

    # Only necessary if history is accessed chronologically, like in trading/backtrading
    if enforce_start:
        too_short = []
        for ticker in df.columns.levels[0]:
            if df[ticker].iloc[0].isnull().values.any():
                too_short.append(ticker)
        if too_short:
            raise ValueError(f'Histories for {too_short} are too short, remove them and start again')

    # If it is not a MultiIndex, make it a MultiIndex: https://stackoverflow.com/questions/40225683/
    if df.columns.nlevels == 1:
        df.columns = pd.MultiIndex.from_product([[tickers], df.columns])
    
    df = df[~df.index.duplicated(keep='last')] # Bug? This drops occasional duplicated rows that show up

    return df

# Pandas bug? Dropped columns stick around in df.columns
# This method offers a simple workaround
def _drop(df, *columns, level):
    df = df.drop(columns=list(columns), level=level)
    df.columns = df.columns.remove_unused_levels()
    return df

def _carrier(ticker, df, functions):
    df = df.droplevel(level=0, axis=1) # Removes ticker level
    
    df = df.dropna() # Drops NaN rows in beginning of series due to start date before IPO

    # Calls functions in technicals.py built from ta library
    for function in functions:
        df = function(df)
    
    df = df.dropna() # Drops NaN rows in beginning of series due to moving averages being undefined
    df.columns = pd.MultiIndex.from_product([[ticker], df.columns]) # Restores ticker level
    return df
def _attach(df, *functions):
    # Splits dataframe by ticker symbol
    df_byticker = df.groupby(axis=1, level=0)

    # Applies ta functions to price histories per ticker symbol, then recombines
    histories = [_carrier(ticker, df_t, functions) for ticker, df_t in df_byticker]
    df = pd.concat(histories, axis=1)

    return df
def apply_config_transforms(df, config_transforms):
    print('Applying transforms on data...', end='')

    transforms = []
    for namestring in config_transforms['tech']:
        transforms.append(tech.TECH_DICT[namestring])
    for ticker in config_transforms['crosscorrelate']:
        transforms.append(moredata.crosscorrelate(ticker))
    for namestring in config_transforms['moredata']:
        transforms.append(moredata.MOREDATA_DICT[namestring])
    df = _attach(df, *transforms)

    drops = []
    for column in config_transforms['drop']:
        drops.append(column)
    df = _drop(df, *drops, level=1)

    print('done')
    
    return df

# Behaves like C sub_str but for dfs and multi-indexes
# Outputs a df smaller than passed size if there is not enough rows
def sub_df(df, index, size):
    end_index = index+size
    if end_index != 0:
        df = df.iloc[index:end_index]
    else: # Special case for negative indexing
        df = df.iloc[index:]
    return df

def count_columns(df, level):
    return len(df.columns.levels[level])

# <Normal Dataframe Functions>

def split(df, ratio=None, date=None):
    length = len(df.index)
    if ratio:
        end_index = int(length*ratio)
    elif date:
        timestamp = pd.Timestamp(date)
        end_index = df.index.get_loc(timestamp, 'nearest')
    else:
        raise ValueError('Specify either a ratio or a date')
    
    training_df = df[:].iloc[0:end_index]
    validation_df = df[:].iloc[end_index:length]
    
    return training_df, validation_df

# Min-max normalization
def normalize(df, exclude_label=False):
    if exclude_label:
        df_input = df.iloc[0:-1]
        df = (df-df_input.min()) / (df_input.max()-df_input.min())
    else:
        df = (df-df.min()) / (df.max()-df.min())
    df = df.replace([np.inf, -np.inf, np.nan], 0)
    return df