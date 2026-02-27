"""
data ingestion of tickers
"""
import os
from datetime import datetime,timedelta
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
import hayai_util as util

def fetch_data(symbol:str,client:StockHistoricalDataClient)->pd.DataFrame:
    request_params = StockBarsRequest(
    symbol_or_symbols=[symbol],
    timeframe=TimeFrame.Day,
    start=datetime.today() - timedelta(days=365*5),
    end=datetime.today() - timedelta(days=1) )
    result = client.get_stock_bars(request_params)
    return result.df

def fetch_data_portfolio(days:int)->bool:
    # if it is from 3:30 PM to 10:00 PM, skip
    now = datetime.now().time()
    if now >= datetime.strptime("15:30:00", "%H:%M:%S").time() and now <= datetime.strptime("22:00:00", "%H:%M:%S").time():
        print("market open, skipping data fetch")
        return False

    apikey = util.context['api_key']
    secret_key = util.context['secret_key']
    client = StockHistoricalDataClient(api_key=apikey,secret_key=secret_key)

    count = len(util.context['symbols'])
    for i, symbol in enumerate(util.context['symbols']):
        filename = os.path.join(util.context['hist_dir'], f"{symbol}.parquet")
        print(f"Fetching data for {symbol} ({i+1}/{count})...")
        df = fetch_data(symbol, client)
        #df['symbol'] = symbol
        df = df.reset_index()
        # remove asset that have less than 1 year of data
        if days > 365 and len(df) < 365:
            continue
        df.to_parquet(filename, index=False)
    return True

def get_actual_position(trading_client: TradingClient)->pd.DataFrame:
    portfolio = trading_client.get_all_positions()
    df = pd.DataFrame([position.__dict__ for position in portfolio])
    df = df[['symbol', 'qty', 'side', 'market_value','current_price']]

    return df