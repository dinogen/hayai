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

def fetch_quotes_yfinance(symbol:str)->pd.DataFrame:
    """fetch historical quotes for a given symbol, with yfinance"""
    data = yf.download(symbol, period="5y", interval="1d")
    data = data.reset_index()
    data['symbol'] = symbol
    data = data[['symbol', 'Date','Open', 'Close', 'High', 'Low', 'Volume']]
    data.columns = ['symbol', 'date', 'open', 'close','high','low', 'volume']
    return data

def fetch_quotes_alpaca(symbol:str,client:StockHistoricalDataClient)->pd.DataFrame:
    """fetch historical quotes for a given symbol, with """
    request_params = StockBarsRequest(
    symbol_or_symbols=[symbol],
    timeframe=TimeFrame.Day,
    start=datetime.today() - timedelta(days=365*5),
    end=datetime.today() - timedelta(days=1) )
    result = client.get_stock_bars(request_params)
    result = result.df.reset_index()
    result = result[['symbol', 'timestamp', 'close', 'volume']]
    result['date'] = result['timestamp'].dt.date
    result = result[['symbol', 'date', 'close', 'volume']]  
    return result

def fetch_quotes(symbol:str,client:StockHistoricalDataClient)->pd.DataFrame:
    """fetch historical quotes for a given symbol, with """
    df = None
    if util.context['data_source'] == 'yfinance':
        df = fetch_quotes_yfinance(symbol)
    if util.context['data_source'] == 'alpaca':
        df = fetch_quotes_alpaca(symbol, client)
    return df

def fetch_quotes_portfolio()->bool:
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
        logger.info("Fetching data from %s for %s (%d/%d)...", util.context['data_source'], symbol, i+1, count)
        df = fetch_quotes(symbol, client)        #df['symbol'] = symbol
        df = df.reset_index(drop=True)
        # remove asset that have less than 1 year of data
        if len(df) < 365:
            continue
        df.to_parquet(filename, index=False)
    return True

def get_latest_price_alpaca(symbols:list[str])->pd.DataFrame:
    """
    Fetch latest trade prices for a list of symbols using Alpaca.
    """
    logger.info("Fetching latest trade prices from Alpaca...")
    client = util.get_stock_historical_data_client()
    request_params = StockLatestTradeRequest(symbol_or_symbols=symbols)
    result = client.get_stock_latest_trade(request_params)
    prices = {k: v.price for k, v in result.items()}
    return pd.DataFrame(list(prices.items()), columns=['symbol', 'price'])

def get_latest_price_yfinance(symbols:list[str])->pd.DataFrame:
    """
    Fetch latest trade prices for a list of symbols using yfinance. 
    Returns a dataframe with two columns: 'symbol' and 'price'.
    """
    logger.info("Fetching latest trade prices from Yahoo Finance...")
    prices = []
    for symbol in symbols:
        if symbol == util.CASH_SYMBOL:
            continue
        ticker = yf.Ticker(symbol)
        if 'postMarketPrice' in ticker.info.keys():
            price = ticker.info['postMarketPrice']
        else:
            price = ticker.info['currentPrice']
        prices.append((symbol, price))
    df = pd.DataFrame(prices, columns=['symbol', 'price'])
        
    return df

def get_latest_price(symbols:list[str])->pd.DataFrame:
    if util.context['data_source'] == 'yfinance':
        return get_latest_price_yfinance(symbols)
    if util.context['data_source'] == 'alpaca':
        return get_latest_price_alpaca(symbols)
    
def get_actual_position_alpaca()->pd.DataFrame:
    """
    Get the actual position from Alpaca. Returns a dataframe with  columns: 'symbol', 'qty_old'.
    """
    logger.info("Fetching actual positions from Alpaca...")
    client = util.get_trading_client()
    portfolio = client.get_all_positions()
    df = pd.DataFrame([position.__dict__ for position in portfolio])
    df['qty'] = df['qty'].astype(float)
    df = df[['symbol', 'qty']]
    df.columns = ['symbol', 'qty_old']
    return df

def get_actual_position_yfinance()->pd.DataFrame:
    """
    Get the actual position from a parquet file. 
    Returns a dataframe with  columns: symbol, qty_old, value_old.
    """
    filename_in = os.path.join(util.context['portfolio_dir'], util.FILE_ACTUAL)
    df = pd.read_parquet(filename_in)
    max_date = df['date'].max()
    df = df[df['date'] == max_date]
    # get latest price for each symbol
    df = df[['symbol', 'qty','value']]
    df.columns = ['symbol', 'qty_old', 'value_old']
    return df

def get_actual_position()->pd.DataFrame:
    """
    Get the actual position.
    Returns a dataframe with  columns: 'symbol', 'qty_old'.
    """
    if util.context['data_source'] == 'yfinance':
        return get_actual_position_yfinance()
    if util.context['data_source'] == 'alpaca':
        return get_actual_position_alpaca()

def get_equity_alpaca()->float:
    """
    Get the equity from Alpaca. Returns a float.
    """
    client = util.get_trading_client()
    account = client.get_account()
    buying_power = float(account.equity)
    return buying_power

def get_equity_yfinance()->float:
    """
    equity_t = CASH_t + Long_t + Short_t
    """
    filename = os.path.join(util.context['portfolio_dir'], "actual_positions.parquet")
    if not os.path.exists(filename):
        logger.warning("Positions file does not exist. Returning 0.")
        return 0.0
    df = pd.read_parquet(filename)
    cash_t = df[df['symbol'] == util.CASH_SYMBOL]['qty_old'].sum()
    long_t = df[df['qty_old'] > 0]['qty_old'].sum()
    short_t = df[df['qty_old'] < 0]['qty_old'].sum()
    equity = cash_t + long_t + short_t
    return equity

def get_equity()->float:
    if util.context['data_source'] == 'yfinance':
        return get_equity_yfinance()
    if util.context['data_source'] == 'alpaca':
        return get_equity_alpaca()

def get_forex()-> pd.DataFrame:
    logger.info("Fetching forex data...")
    symbols = ['GBPUSD=X', 'EURUSD=X', 'USDJPY=X', 'USDCAD=X', 'USDCHF=X', 'AUDUSD=X', 'NZDUSD=X','GC=F','BZ=F','CNYUSD=X']
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

