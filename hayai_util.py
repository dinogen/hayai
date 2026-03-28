"""
This module provides utility functions for trading operations,
including context creation for portfolio management.
It reads configuration settings from a specified directory and
prepares the necessary information for trading activities."""
import configparser
from datetime import datetime,date,timedelta
import os
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient

CASH_SYMBOL = 'MYCASH'
FILE_FEATURES = 'f001_features.parquet'
FILE_PREDICTIONS = 'f002_predictions.parquet'
FILE_WEIGHTS = 'f003_weights.parquet'
FILE_POSITION = 'f004_position.parquet'
FILE_POSITION_NEW = 'f005_position_new.parquet'
FILE_POSITION_NEW_QTY = 'f006_position_new_qty.parquet'
FILE_ORDERS = 'f007_orders.parquet'
FILE_ACTUAL = 'f008_actual.parquet'

def create_context(portfolio_id:str)->dict[str, any]:
    global context
    portfolio_dir = os.path.join('data', portfolio_id)
    hist_dir = os.path.join(portfolio_dir, 'hist')
    if not os.path.exists(portfolio_dir):
        raise FileNotFoundError(f"Portfolio directory '{portfolio_dir}' does not exist.")
    if not os.path.exists(os.path.join(portfolio_dir, 'portfolio.csv')):
        raise FileNotFoundError(f"Portfolio file '{os.path.join(portfolio_dir, 'portfolio.csv')}' does not exist.")
    if not os.path.exists(os.path.join(portfolio_dir, 'conf.ini')):
        raise FileNotFoundError(f"Configuration file '{os.path.join(portfolio_dir, 'conf.ini')}' does not exist.")
    if not os.path.exists(hist_dir):
        os.mkdir(os.path.join(portfolio_dir, 'hist'))
    df = pd.read_csv(os.path.join(portfolio_dir, 'portfolio.csv'),keep_default_na=False)
    df = df[df['Country'] != '']
    df = df[df['Sector']  != '']
    symbols = df['Symbol'].tolist()

    # global secret.ini
    secret_global = configparser.ConfigParser()
    secret_global.read('secret.ini')
    telegram_api_id = secret_global.get('telegram', 'api_id')
    telegram_api_hash = secret_global.get('telegram', 'api_hash')
    telegram_bot_token = secret_global.get('telegram', 'bot_token')
    telegram_chat_id = secret_global.get('telegram', 'chat_id')

    # portfolio conf.ini
    conf_portfolio = configparser.ConfigParser()
    conf_portfolio.read(os.path.join(portfolio_dir, 'conf.ini'))
    volatility_window = conf_portfolio.getint('features', 'volatility_window', fallback=20)
    target_return_days = conf_portfolio.getint('features', 'target_return_days', fallback=5)
    mean_window = conf_portfolio.getint('features', 'mean_window', fallback=20)
    epochs = conf_portfolio.getint('training', 'epochs', fallback=20)
    batch_size = conf_portfolio.getint('training', 'batch_size', fallback=64)
    learning_rate = conf_portfolio.getfloat('training', 'learning_rate', fallback=0.001)
    validation_split = conf_portfolio.getfloat('training', 'validation_split', fallback=0.2)
    n_long = conf_portfolio.getint('portfolio', 'n_long', fallback=5)
    n_short = conf_portfolio.getint('portfolio', 'n_short', fallback=5)
    risk_percentage = conf_portfolio.getfloat('portfolio', 'risk_percentage', fallback=0.8)
    qty_diff_perc_min = conf_portfolio.getfloat('portfolio', 'qty_diff_perc_min', fallback=0.2)
    data_source = conf_portfolio.get('features', 'data_source', fallback='yfinance')
    if portfolio_id.startswith('model_'):
        model_name = portfolio_id
    else:
        model_name = conf_portfolio.get('predictions', 'model', fallback='model')

    # model conf.ini
    conf_model = configparser.ConfigParser()
    model_dir = os.path.join("data", model_name)
    conf_model.read(os.path.join(model_dir, 'conf.ini'))
    clip_min = conf_model.getfloat('predictions', 'clip_min', fallback=-5)
    clip_max = conf_model.getfloat('predictions', 'clip_max', fallback=5)
    label_min = conf_model.getfloat('predictions', 'label_min')
    label_max = conf_model.getfloat('predictions', 'label_max')
    forex = conf_model.get('features', 'forex', fallback='')
    forex = [s.strip() for s in forex.split(',')] if forex else []
    indexes = conf_model.get('features', 'indexes', fallback='')
    indexes = [s.strip() for s in indexes.split(',')] if indexes else []

    # portfolio secret.ini
    secret_portfolio = configparser.ConfigParser()
    secret_portfolio.read(os.path.join(portfolio_dir, 'secret.ini'))
    api_key = secret_portfolio.get('portfolio', 'api_key')
    secret_key = secret_portfolio.get('portfolio', 'secret_key')


    context = {'api_key': api_key,
                'secret_key': secret_key,
                'portfolio_dir': portfolio_dir,
                'hist_dir': hist_dir,
                'symbols': symbols,
                'portfolio_id': portfolio_id,
                'volatility_window': volatility_window,
                'target_return_days': target_return_days,
                'mean_window': mean_window,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'validation_split': validation_split,
                'clip_min': clip_min,
                'clip_max': clip_max,
                'n_long': n_long,
                'n_short': n_short,
                'risk_percentage': risk_percentage,
                'qty_diff_perc_min': qty_diff_perc_min,
                'model_dir': model_dir,
                'telegram_api_id': telegram_api_id,
                'telegram_api_hash': telegram_api_hash,
                'telegram_bot_token': telegram_bot_token,
                'telegram_chat_id': telegram_chat_id,
                'chat:id': telegram_chat_id,
                'data_source': data_source,
                'label_min': label_min,
                'label_max': label_max,
                'forex': forex,
                'indexes': indexes}
    return context

def save_normalization_params(label_min:float, label_max:float)->bool:
    conf_portfolio = configparser.ConfigParser()
    conf_portfolio.read(os.path.join(context['portfolio_dir'], 'conf.ini'))

    conf_portfolio.set('predictions', 'label_min', str(label_min))
    conf_portfolio.set('predictions', 'label_max', str(label_max))

    with open(os.path.join(context['portfolio_dir'], 'conf.ini'), 'w',encoding='utf-8') as configfile:
        conf_portfolio.write(configfile)

    return True

def get_trading_client()->TradingClient:
    apikey = context['api_key']
    secret_key = context['secret_key']
    trading_client = TradingClient(api_key=apikey,secret_key=secret_key,paper=True)
    return trading_client

def get_stock_historical_data_client()->StockHistoricalDataClient:
    apikey = context['api_key']
    secret_key = context['secret_key']
    client = StockHistoricalDataClient(api_key=apikey,secret_key=secret_key)
    return client

def get_report_name()->str:
    filename = os.path.join(context['portfolio_dir'], f"report_{context['portfolio_id']}.html")
    return filename

def create_report(df_new_qty:pd.DataFrame)->bool:
    """Create a html file with the current positions, including symbol, quantity, price and value.
    Return true in success, false otherwise."""
    filename_out =  get_report_name()
    df = df_new_qty[df_new_qty['qty_new'] != 0]
    long = 0.0
    short = 0.0
    cash = 0.0
    html = f"""<h1>Portfolio Position Report at {date.today()}</h1>
    <h2>Portfolio: {context['portfolio_id']}</h2>
    <table><tr><th>Symbol</th><th>Qty</th><th>Price</th><th>Value</th></tr>"""
    for index, row in df.iterrows():
        symbol = row['symbol']
        if symbol == CASH_SYMBOL:
            cash = row['value_new']
            continue
        qty = row['qty_new']
        price = row['price']
        value = row['value_new']
        if qty > 0:
            long += value
        else:
            short += value
        html += f"<tr><td>{symbol}</td><td>{qty}</td><td>${price:.2f}</td><td>${value:.2f}</td></tr>"
    html += f"<tr><td colspan='3'>Total Long</td><td>${long:.2f}</td></tr>"
    html += f"<tr><td colspan='3'>Total Short</td><td>${short:.2f}</td></tr>"
    html += f"<tr><td colspan='3'>Net Position</td><td>${long + short:.2f}</td></tr>"
    html += f"<tr><td colspan='3'>Cash</td><td>${cash:.2f}</td></tr>"
    html += f"<tr><td colspan='3'>Total Portfolio Value</td><td>${long + short + cash:.2f}</td></tr>"
    html += "</table>"
    with open(filename_out, 'w', encoding='utf-8') as f:
        f.write(html)
    return True
