"""
This module provides utility functions for trading operations, 
including context creation for portfolio management. 
It reads configuration settings from a specified directory and 
prepares the necessary information for trading activities."""
import configparser
import os
import pandas as pd
def create_context(portfolio_id:str)->dict[str, any]:
    global context
    config = configparser.ConfigParser()
    config.read('conf.ini')
    api_key = config.get('alpaca', 'api_key')
    secret_key = config.get('alpaca', 'secret_key')
    
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

    conf_portfolio = configparser.ConfigParser()
    conf_portfolio.read(os.path.join(portfolio_dir, 'conf.ini'))
    volatility_window = conf_portfolio.getint('features', 'volatility_window')
    target_return_days = conf_portfolio.getint('features', 'target_return_days')
    mean_window = conf_portfolio.getint('features', 'mean_window')
    epochs = conf_portfolio.getint('training', 'epochs')
    batch_size = conf_portfolio.getint('training', 'batch_size')
    learning_rate = conf_portfolio.getfloat('training', 'learning_rate')
    validation_split = conf_portfolio.getfloat('training', 'validation_split')
    label_min = conf_portfolio.getfloat('predictions', 'label_min')
    label_max = conf_portfolio.getfloat('predictions', 'label_max')
    n_long = conf_portfolio.getint('portfolio', 'n_long')
    n_short = conf_portfolio.getint('portfolio', 'n_short')
    clip_min = conf_portfolio.getfloat('predictions', 'clip_min')
    clip_max = conf_portfolio.getfloat('predictions', 'clip_max')
    risk_percentage = conf_portfolio.getfloat('portfolio', 'risk_percentage')   

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
                'risk_percentage': risk_percentage}
    return context

def save_normalization_params(label_min:float, label_max:float)->bool:
    conf_portfolio = configparser.ConfigParser()
    conf_portfolio.read(os.path.join(context['portfolio_dir'], 'conf.ini'))

    conf_portfolio.set('predictions', 'label_min', str(label_min))
    conf_portfolio.set('predictions', 'label_max', str(label_max))

    with open(os.path.join(context['portfolio_dir'], 'conf.ini'), 'w',encoding='utf-8') as configfile:
        conf_portfolio.write(configfile)

    return True