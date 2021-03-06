import pandas as pd
import yfinance as yf

def crosscorrelate(ticker):
    df_ticker = yf.Ticker(ticker).history(period='max')
    df_ticker.columns = [ticker+' '+column for column in df_ticker.columns]
    def crosscorrelate_ticker(df):
        df = pd.concat([df, df_ticker], axis=1)
        return df
    return crosscorrelate_ticker

def day_of_week(df):
    series_day_of_week = df.index.dayofweek

    # As a number
    df['day_of_week'] = series_day_of_week
    # One-hot encoding
    df['mon_oh'] = (series_day_of_week == 0).astype(int)
    df['tues_oh'] = (series_day_of_week == 1).astype(int)
    df['wed_oh'] = (series_day_of_week == 2).astype(int)
    df['thurs_oh'] = (series_day_of_week == 3).astype(int)
    df['fri_oh'] = (series_day_of_week == 4).astype(int)
    df['sat_oh'] = (series_day_of_week == 5).astype(int)
    df['sun_oh'] = (series_day_of_week == 6).astype(int)

    return df

MOREDATA_DICT = {
    'day_of_week': day_of_week
}