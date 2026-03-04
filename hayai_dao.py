"""
data ingestion of tickers
"""
import os
from datetime import datetime,timedelta
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
import hayai_util as util

def fetch_quotes(symbol:str,client:StockHistoricalDataClient)->pd.DataFrame:
    """fetch historical quotes for a given symbol, with """
    request_params = StockBarsRequest(
    symbol_or_symbols=[symbol],
    timeframe=TimeFrame.Day,
    start=datetime.today() - timedelta(days=365*5),
    end=datetime.today() - timedelta(days=1) )
    result = client.get_stock_bars(request_params)
    return result.df

def fetch_quotes_portfolio(days:int)->bool:
    # if it is from 3:30 PM to 10:00 PM, skip
    now = datetime.now().time()
    if now >= datetime.strptime("15:30:00", "%H:%M:%S").time() and now <= datetime.strptime("22:00:00", "%H:%M:%S").time():
        print("market open, skipping data fetch")
        return False

    apikey = util.context['api_key']
    secret_key = util.context['secret_key']
    client:StockHistoricalDataClient = util.get_stock_historical_data_client()

    count = len(util.context['symbols'])
    for i, symbol in enumerate(util.context['symbols']):
        filename = os.path.join(util.context['hist_dir'], f"{symbol}.parquet")
        print(f"Fetching data for {symbol} ({i+1}/{count})...")
        df = fetch_quotes(symbol, client)
        #df['symbol'] = symbol
        df = df.reset_index()
        # remove asset that have less than 1 year of data
        if days > 365 and len(df) < 365:
            continue
        df.to_parquet(filename, index=False)
    return True

def get_latest_trade_price(symbols:list[str])->float:
    client = util.get_stock_historical_data_client()
    request_params = StockLatestTradeRequest(symbol_or_symbols=symbols)
    result = client.get_stock_latest_trade(request_params)
    prices = {k: v.price for k, v in result.items()}
    return pd.DataFrame(list(prices.items()), columns=['symbol', 'price'])

def get_actual_position()->pd.DataFrame:
    # Get the actual positions from the trading client
    client = util.get_trading_client()
    portfolio = client.get_all_positions()
    df = pd.DataFrame([position.__dict__ for position in portfolio])
    df['qty'] = df['qty'].astype(float)
    df = df[['symbol', 'qty', 'side']]
    df.columns = ['symbol', 'qty_old', 'side_old']
    return df

def get_account_info()->pd.DataFrame:
    client = util.get_trading_client()
    account = client.get_account()
    return account
