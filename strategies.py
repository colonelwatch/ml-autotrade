import numpy as np
import torch

import database as db
import neural

# Not the same as target_tickers, this holds the stock universe, and predictions are generated for all stocks
STOCK_UNIVERSE = db.load_txt('./stock_lists/stock_universe.txt')
trading_stocks = db.load_txt('./stock_lists/trading_tickers.txt')

# <Classic Trading Strategies>

# Liquidates all holdings and splits total buying power equally between passed tickers, then does nothing
bought_once = False
def buy_and_hold(portfolio, market, target_tickers):
    global bought_once
    if not bought_once:
        holdings = portfolio.holdings
        for ticker in holdings:
            if holdings[ticker] != 0: # Never send orders of quantity 0
                portfolio.market_sell(ticker, holdings[ticker])
        cash = portfolio.cash
        cash_split = cash/len(target_tickers)
        for ticker in target_tickers:
            portfolio.market_buy(ticker, cash_split/market.price(ticker))
        bought_once = True
    else:
        pass
buy_and_hold.name = 'Buy and Hold'
buy_and_hold.target_tickers = trading_stocks

# <Neural Net Initialization>

state = torch.load('model.pt')
n_input = state['n_input']
model = neural.LSTM(n_input) # Resets model with a batch size of target list length
model.load_state_dict(state['model'])
model.eval()
print('LSTM loaded')

# <Neural Net Strategies>

# TODO: Implement multiprocessing or further optimization.
def predictreturns(df, target_tickers):
    tensor_length = neural.INTERVAL_TENSOR_SIZE
    
    df = df[target_tickers]
    df_slice = df.iloc[-tensor_length:]
    
    # Pulls the close data from the dataframe to get the parameters to undo normalization later
    df_slice_close = df_slice.xs('Close', level=1, axis=1)
    mn = df_slice_close.min().values
    mx = df_slice_close.max().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        arr = db.normalize(df_slice.values)
        arr = np.nan_to_num(arr, copy=False, nan=0, posinf=0, neginf=0)
    # Converts dataframe into tensor with dimensions (batch/tickers, length/time, features/indicators)
    input_tensor = torch.tensor(arr).reshape(tensor_length, -1, n_input).permute(1, 0, 2).float()
    
    output = model(input_tensor)
    # Pulls the last row of closing prices and undoes normailization
    last = df_slice_close.iloc[-1].values
    output = output.detach().numpy()*(mx-mn)+mn
    returns = (output-last)/last
    
    returns = returns.flatten().tolist() # flatten() seems to be necessary here
    returns = {ticker:return_val for ticker, return_val in zip(target_tickers, returns)}

    return returns

# <Neural Net Based Strategies>

def neural_split_value(portfolio, market, target_tickers):
    holdings = portfolio.holdings
    df = market.latest_data
    predictions = predictreturns(df, target_tickers)
    
    wanted_tickers = []
    for ticker in predictions:
        if predictions[ticker] > 0:
            wanted_tickers.append(ticker)
        elif predictions[ticker] <= 0 and holdings[ticker] > 0:
            portfolio.market_sell(ticker, holdings[ticker]) # Simply eliminate expected losing tickers immediately
    
    if wanted_tickers:
        value = portfolio.cash
        for ticker in wanted_tickers:
            value += holdings[ticker]*market.price(ticker)
        value_split = value/len(wanted_tickers)
        
        to_expand = {}
        to_shrink = {}
        for ticker in wanted_tickers:
            quantity_change = int(value_split/market.price(ticker)) - holdings[ticker]
            if quantity_change > 0:
                to_expand[ticker] = quantity_change
            elif quantity_change < 0:
                to_shrink[ticker] = -quantity_change
        
        for ticker in to_shrink:
            portfolio.market_sell(ticker, to_shrink[ticker])
        for ticker in to_expand:
            portfolio.market_buy(ticker, to_expand[ticker])
neural_split_value.name = 'Neural Value Split'
neural_split_value.target_tickers = trading_stocks

def neural_weighted_split_value(portfolio, market, target_tickers):
    holdings = portfolio.holdings
    df = market.latest_data
    predictions = predictreturns(df, target_tickers)
    
    return_sum = 0
    wanted_tickers = []
    for ticker in predictions:
        if predictions[ticker] > 0:
            return_sum += predictions[ticker]
            wanted_tickers.append(ticker)
        elif predictions[ticker] <= 0 and holdings[ticker] > 0:
            portfolio.market_sell(ticker, holdings[ticker]) # Simply eliminate expected losing tickers immediately
    
    if wanted_tickers:
        value = portfolio.cash
        for ticker in wanted_tickers:
            value += holdings[ticker]*market.price(ticker)
        if return_sum != 0:
            value_split = {ticker:value*predictions[ticker]/return_sum for ticker in wanted_tickers}
        else:
            value_split = {ticker:0 for ticker in wanted_tickers}
        
        to_expand = {}
        to_shrink = {}
        for ticker in wanted_tickers:
            quantity_change = int(value_split[ticker]/market.price(ticker)) - holdings[ticker]
            if quantity_change > 0:
                to_expand[ticker] = quantity_change
            elif quantity_change < 0:
                to_shrink[ticker] = -quantity_change
        
        for ticker in to_shrink:
            portfolio.market_sell(ticker, to_shrink[ticker])
        for ticker in to_expand:
            portfolio.market_buy(ticker, to_expand[ticker])
neural_weighted_split_value.name = 'Neural Weighted Value Split'
neural_weighted_split_value.target_tickers = trading_stocks

def neural_squareweighted_split_value(portfolio, market, target_tickers):
    holdings = portfolio.holdings
    df = market.latest_data
    predictions = predictreturns(df, target_tickers)

    # Square the percentage changes to emphasize higher predictions in the portfolio
    for ticker, prediction in predictions.items():
        if prediction > 0:
            predictions[ticker] = prediction**2
        else:
            predictions[ticker] = prediction
    
    return_sum = 0
    wanted_tickers = []
    for ticker in predictions:
        if predictions[ticker] > 0:
            return_sum += predictions[ticker]
            wanted_tickers.append(ticker)
        elif predictions[ticker] <= 0 and holdings[ticker] > 0:
            portfolio.market_sell(ticker, holdings[ticker]) # Simply eliminate expected losing tickers immediately
    
    if wanted_tickers:
        value = portfolio.cash
        for ticker in wanted_tickers:
            value += holdings[ticker]*market.price(ticker)
        if return_sum != 0:
            value_split = {ticker:value*predictions[ticker]/return_sum for ticker in wanted_tickers}
        else:
            value_split = {ticker:0 for ticker in wanted_tickers}
        
        to_expand = {}
        to_shrink = {}
        for ticker in wanted_tickers:
            quantity_change = int(value_split[ticker]/market.price(ticker)) - holdings[ticker]
            if quantity_change > 0:
                to_expand[ticker] = quantity_change
            elif quantity_change < 0:
                to_shrink[ticker] = -quantity_change
        
        for ticker in to_shrink:
            portfolio.market_sell(ticker, to_shrink[ticker])
        for ticker in to_expand:
            portfolio.market_buy(ticker, to_expand[ticker])
neural_squareweighted_split_value.name = 'Neural Square-Weighted Value Split'
neural_squareweighted_split_value.target_tickers = trading_stocks

strategies = [buy_and_hold, neural_split_value, neural_weighted_split_value, neural_squareweighted_split_value]
main_strategy = neural_weighted_split_value

print('Strategies generated')