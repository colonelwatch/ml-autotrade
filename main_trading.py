import json
import time
import alpaca_trade_api as tradeapi

import database as db
import strategies as st

with open('config.json', 'r') as infile:
    config = json.load(infile)
    CONFIG_TRANSFORMS = config['transforms']
    API_KEY = config['alpaca']['key']
    API_SECRET = config['alpaca']['secret']
    API_URL = config['alpaca']['url']
    SECONDS_BEFORE_CLOSE = config['seconds_before_close']

class liveMarket:
    def __init__(self, API_KEY, API_SECRET, API_URL):
        self.broker = tradeapi.REST(API_KEY, API_SECRET, base_url=API_URL)
    def price(self, ticker):
        trade = self.broker.get_last_trade(ticker)
        return float(trade.price)
    def latest_data(self): # Returns 1 year of data, but could be a different period if needed
        tickers = st.STOCK_UNIVERSE
        df = db.download_history(tickers, enforce_start=True, period='1y', interval='1d')
        df = db.apply_config_transforms(df, CONFIG_TRANSFORMS) # Adds more data (technicals, etc)
        df = df.dropna().reset_index(drop=True) # Eliminates all rows without records for all tickers
        return df
    def is_open(self):
        clock = self.broker.get_clock()
        return clock.is_open
    def get_seconds_to_close(self):
        clock = self.broker.get_clock()
        timedelta = clock.next_close-clock.timestamp
        return timedelta.total_seconds()
    def get_seconds_to_open(self):
        clock = self.broker.get_clock()
        timedelta = clock.next_open-clock.timestamp
        return timedelta.total_seconds()

market = liveMarket(API_KEY, API_SECRET, API_URL)

class livePortfolio: # Wraps market into a form like virtualPortfolio for strategies.py to access
    def __init__(self):
        pass
    @property
    def cash(self):
        return float(market.broker.get_account().cash)
    @property
    def holdings(self):
        holdings = {ticker:0 for ticker in st.STOCK_UNIVERSE}
        list_positions = market.broker.list_positions()
        for position in list_positions:
            holdings[position.symbol] = int(position.qty)
        return holdings
    def market_buy(self, ticker, quantity):
        order = market.broker.submit_order(
            symbol = ticker,
            side = 'buy',
            type = 'market',
            qty = quantity,
            time_in_force = 'day'
        )
        return order.id
    def market_sell(self, ticker, quantity):
        order = market.broker.submit_order(
            symbol = ticker,
            side = 'sell',
            type = 'market',
            qty = quantity,
            time_in_force = 'day'
        )
        return order.id
        
portfolio = livePortfolio()
strategy = st.main_strategy

if not market.is_open():
    time.sleep(market.get_seconds_to_open())
while(True):
    time.sleep(market.get_seconds_to_close()-SECONDS_BEFORE_CLOSE)
    print('Running strategy and executing orders... ', end='')
    strategy(portfolio, market, strategy.target_tickers)
    print('Done')
    time.sleep(market.get_seconds_to_close())
    time.sleep(market.get_seconds_to_open())