import ta

def bollinger(df):
    # Initialize Bollinger Bands function
    indicator_bb = ta.volatility.BollingerBands(close=df['Close'])

    # Add Bollinger Bands features
    df['bb_avg'] = indicator_bb.bollinger_mavg()
    df['bb_high'] = indicator_bb.bollinger_hband()
    df['bb_low'] = indicator_bb.bollinger_lband()

    return df

def rsi(df):
    # Initialize RSI function
    indicator_rsi = ta.momentum.RSIIndicator(close=df['Close'])

    # Add RSI features
    df['rsi'] = indicator_rsi.rsi()

    return df

def macd(df):
    # Initialize MACD function
    indicator_macd = ta.trend.MACD(close=df['Close'])

    # Add MACD features
    df['macd'] = indicator_macd.macd()
    df['macd_signal'] = indicator_macd.macd_signal()

    return df

def obv(df):
    # Initialize OBV function
    indicator_obv = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])

    # Add OBV features
    df['obv'] = indicator_obv.on_balance_volume()

    return df

# Relates names in config.json to functions
TECH_DICT = {
    'bollinger': bollinger,
    'rsi': rsi,
    'macd': macd,
    'obv': obv
}