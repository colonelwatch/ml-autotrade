import pandas as pd
import yfinance as yf

def crosscorrelate(ticker):
    df_ticker = yf.Ticker(ticker).history(period='max')
    df_ticker.columns = [ticker+' '+column for column in df_ticker.columns]
    def crosscorrelate_ticker(df):
        df = pd.concat([df, df_ticker], axis=1)
        return df
    return crosscorrelate_ticker