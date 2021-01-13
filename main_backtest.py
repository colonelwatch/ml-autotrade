import json
import matplotlib.pyplot as plt

import database as db
import strategies as st

with open('config.json', 'r') as infile:
    config = json.load(infile)
    CONFIG_TRANSFORMS = config['transforms']
    DF_LENGTH = config['df_slice_length'] # shouldn't be less than neural.INTERVAL_TENSOR_SIZE

backtest_stocks = st.STOCK_UNIVERSE

# The start date is the same as the training/validation split date, so it's guaranteed overlap free
df = db.download_history(backtest_stocks, cache_name='backtest_stocks', enforce_start=True, start='2018-01-02', interval='1d')

df = db.apply_config_transforms(df, CONFIG_TRANSFORMS) # Adds more data (technicals, etc)

class virtualMarket:
    def __init__(self, df):
        self.data = df
        self.index = 1
    def price(self, ticker):
        return self.data[ticker]['Close'].iloc[self.index+DF_LENGTH-1] # Gets last price in series
    def latest_data(self):
        df_slice = db.sub_df(self.data, self.index, DF_LENGTH) # Generates 'latest' market data
        return df_slice
    def next_day(self):
        self.index += 1
        # If there are not enough days in the history to fill a tensor, return False
        if self.index+DF_LENGTH <= len(self.data.index):
            return True
        else:
            return False

df = df.dropna().reset_index(drop=True) # Eliminates all rows without records for all tickers
vMarket = virtualMarket(df)

class virtualPortfolio:
    def __init__(self, cash):
        self.holdings = {}
        self.cash = cash
        self.ConditionalOrders = []
        self.value_history = []
    def execute_order(self, side, ticker, quantity, price):
        if quantity <= 0:
            raise ValueError('Quantity cannot be 0 or negative')
        if side == 'buy':
            self.cash -= price*quantity
            self.holdings[ticker] += quantity
        elif side == 'sell':
            self.cash += price*quantity
            self.holdings[ticker] -= quantity
    def market_buy(self, ticker, quantity):
        self.execute_order('buy', ticker, quantity, vMarket.price(ticker))
    def market_sell(self, ticker, quantity):
        self.execute_order('sell', ticker, quantity, vMarket.price(ticker))
    def updateValueHistory(self):
        holdings = self.holdings
        value = self.cash
        if int(value) < 0 :
            raise AssertionError(f'Negative cash balance of {value} detected!')
        for ticker in holdings:
            if holdings[ticker] < 0:
                raise AssertionError(f'Ticker {ticker} is at a negative quantity.')
            value += holdings[ticker]*vMarket.price(ticker)
        self.value_history.append(value)

portfolios = {}
for strategy in st.strategies:
    portfolio = virtualPortfolio(27000)
    portfolio.holdings = {ticker:0 for ticker in strategy.target_tickers} # Initializes portfolio with 0s
    portfolios[strategy] = portfolio

def day_actions():
    for strategy, portfolio in portfolios.items():
        strategy(portfolio, vMarket, strategy.target_tickers)
        portfolio.updateValueHistory()

def plotStrategies():
    for strategy, portfolio in portfolios.items():
        plt.plot(portfolio.value_history, label=strategy.name)
    plt.legend()
    plt.show()

print('Calculating virtual market results, may take a few minutes...')

day_actions()
while(vMarket.next_day()):
    day_actions()
plotStrategies()