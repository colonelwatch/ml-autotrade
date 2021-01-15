# ml-autotrade

This is a set of Python scripts that provide a framework for training neural networks to predict stock prices, developing 
trading strategies that use these predictions, and applying these strategies on the real stock market through Alpaca. It 
is based on PyTorch.

## Usage/Explanation

All dependencies can by retrieved by calling `pip install -r requirements.txt` or `pip3 install -r requirements.txt`.

All functionality comes from executing four scripts:
1) `main_training.py`: Trains the neural network.
2) `main_eval.py`: Evaluates the neural network by plotting the known change in price against the predicted change.
3) `main_backtest.py`: Evaluates strategies using the neural network by backtesting against a given stock universe.
4) `main_trading.py`: Executes one strategy using the neural network on the real stock market.

There are also five configurable files, where a working example is already configured:
1) `config.json`: Global constants, including augmentation (using technical analysis indicators), training, and model 
    parameters along with the Alpaca API key.
2) `strategies.py`: Abstracted Python functions that outlines strategies based on neural network predictions, portfolio 
    holdings, and portfolio cash-on-hand. Raw "latest" market data can also be accessed.
3) `stock_lists/training_tickers.txt`: Tickers whose histories with be trained on by the neural network.
5) `stock_lists/trading_tickers.txt`: Tickers that all of the example strategies will trade.
4) `stock_lists/stock_universe.txt`: Tickers whose histories `strategies.py` can access. *All traded tickers* must be in 
    this list, and all tickers in this list must have a history that extend past the backtesting start date.

In addition, there are four other functional files:
1) `database.py`: Functions that retrieve, access, and modify price history databases.
2) `technicals.py`: Functions that calculate and attach technical analysis indicators. These can also be called in 
    `strategies.py` to make ordinary trading decisions.
3) `moredata.py`: Functions that retrieve and attach additional data.
4) `neural.py`: Definition of the neural network and the PyTorch `Dataset`.