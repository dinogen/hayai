"""
This module provides utility functions for trading operations, 
including context creation for portfolio management. 
It reads configuration settings from a specified directory and 
prepares the necessary information for trading activities."""
import configparser
import os
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient


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
    df = pd.read_csv(os.path.join(portfolio_dir, 'portfolio.csv'))
    symbols = df['Symbol'].tolist()

    conf_global = configparser.ConfigParser()
    conf_global.read('conf.ini')
    telegram_api_id = conf_global.get('telegram', 'api_id')
    telegram_api_hash = conf_global.get('telegram', 'api_hash')
    telegram_bot_token = conf_global.get('telegram', 'bot_token')
    telegram_chat_id = conf_global.get('telegram', 'chat_id')

    conf_portfolio = configparser.ConfigParser()
    conf_portfolio.read(os.path.join(portfolio_dir, 'conf.ini'))
    volatility_window = conf_portfolio.getint('features', 'volatility_window')
    target_return_days = conf_portfolio.getint('features', 'target_return_days')
    mean_window = conf_portfolio.getint('features', 'mean_window')
    epochs = conf_portfolio.getint('training', 'epochs')
    batch_size = conf_portfolio.getint('training', 'batch_size')
    learning_rate = conf_portfolio.getfloat('training', 'learning_rate')
    validation_split = conf_portfolio.getfloat('training', 'validation_split')
    n_long = conf_portfolio.getint('portfolio', 'n_long')
    n_short = conf_portfolio.getint('portfolio', 'n_short')
    api_key = conf_portfolio.get('portfolio', 'api_key')
    secret_key = conf_portfolio.get('portfolio', 'secret_key')
    risk_percentage = conf_portfolio.getfloat('portfolio', 'risk_percentage')   
    qty_diff_perc_min = conf_portfolio.getfloat('portfolio', 'qty_diff_perc_min')

    conf_model = configparser.ConfigParser()
    model_dir = os.path.join("data","model")
    conf_model.read(os.path.join(model_dir, 'conf.ini'))
    label_min = conf_model.getfloat('predictions', 'label_min')
    label_max = conf_model.getfloat('predictions', 'label_max')
    clip_min = conf_model.getfloat('predictions', 'clip_min')
    clip_max = conf_model.getfloat('predictions', 'clip_max')


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
                'label_min': label_min,
                'label_max': label_max,
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
                'chat:id': telegram_chat_id}
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
