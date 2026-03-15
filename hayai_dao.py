"""
data ingestion of tickers
"""
import os
from datetime import datetime,timedelta
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import yfinance as yf
import hayai_util as util

import hayai_log
logger = hayai_log.create_logger(__name__)

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
    """ fetch historical quotes for all symbols in the portfolio, and save to parquet. """
    client:StockHistoricalDataClient = util.get_stock_historical_data_client()
    count = len(util.context['symbols'])
    for i, symbol in enumerate(util.context['symbols']):
        filename = os.path.join(util.context['hist_dir'], f"{symbol}.parquet")
        if os.path.exists(filename):
            mtime = datetime.fromtimestamp(os.path.getmtime(filename))
            atime = datetime.fromtimestamp(os.path.getatime(filename))
            ctime = datetime.fromtimestamp(os.path.getctime(filename))
            mytime = max(mtime, atime, ctime)
            if datetime.now() - mytime < timedelta(hours=4):
                logger.info("Skipping %s (%d/%d), file is recent (< 4h)", symbol, i+1, count)
                continue
        logger.info("Fetching data for %s (%d/%d)...", symbol, i+1, count)
        df = fetch_quotes(symbol, client)        #df['symbol'] = symbol
        df = df.reset_index()
        # remove asset that have less than 1 year of data
        if days > 365 and len(df) < 365:
            continue
        df.to_parquet(filename, index=False)
    return True

def get_latest_trade_price(symbols:list[str])->float:
    logger.info("Fetching latest trade prices...")
    client = util.get_stock_historical_data_client()
    request_params = StockLatestTradeRequest(symbol_or_symbols=symbols)
    result = client.get_stock_latest_trade(request_params)
    prices = {k: v.price for k, v in result.items()}
    return pd.DataFrame(list(prices.items()), columns=['symbol', 'price'])

def get_actual_position()->pd.DataFrame:
    # Get the actual positions from the trading client
    logger.info("Fetching actual positions...")
    client = util.get_trading_client()
    portfolio = client.get_all_positions()
    df = pd.DataFrame([position.__dict__ for position in portfolio])
    df['qty'] = df['qty'].astype(float)
    df = df[['symbol', 'qty', 'side']]
    df.columns = ['symbol', 'qty_old', 'side_old']
    return df

def get_account_info()->pd.DataFrame:
    logger.info("Fetching account info...")
    client = util.get_trading_client()
    account = client.get_account()
    return account

def get_forex()-> pd.DataFrame:
    logger.info("Fetching forex data...")
    symbols = ['GBPUSD=X', 'EURUSD=X', 'USDJPY=X', 'USDCAD=X', 'USDCHF=X', 'AUDUSD=X', 'NZDUSD=X','GC=F','BZ=F']
    data = yf.download(symbols, period="5y", interval="1d")
    df = data['Close']
    df.columns = [col.split('=')[0] for col in df.columns]
    df['date'] = pd.to_datetime(df.index).date
    df = df.reset_index(drop=True)
    return df

def get_index()-> pd.DataFrame:
    logger.info("Fetching index data...")
    symbols = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX1D']
    data = yf.download(symbols, period="5y", interval="1d")
    df = data['Close']
    df.columns = [col.split('=')[0] for col in df.columns]
    df['date'] = pd.to_datetime(df.index).date
    df = df.reset_index(drop=True)
    return df

