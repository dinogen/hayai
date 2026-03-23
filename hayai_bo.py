""" This module contains the main logic for the backtesting
and trading of the portfolio. It includes functions to add features to the data,
apply the model, define weights, build new positions,
define new quantities, and execute trades. """
import os
from datetime import date
import pandas as pd
import numpy as np
import hayai_util as util
import hayai_dao as dao
import hayai_trade as trade

import hayai_log
logger = hayai_log.create_logger(__name__)

def add_time_features(df:pd.DataFrame)->pd.DataFrame:
    """ Add time-related features to the dataframe, and return it. """
    df['timestamp'] = pd.to_datetime(df['date'])
    df['date'] = df['timestamp'].dt.date
    df = df.set_index('date')
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['time_since_high'] = 0 # set initial value to 0, then update it iterating over the dataframe
    max_close = -1
    days_since_high = 0
    for i,row in df.iterrows():
        if row['close'] > max_close:
            max_close = row['close']
            days_since_high = 0
            # the value is already 0
        else:
            days_since_high += 1
            df.loc[i,'time_since_high'] = days_since_high
    df.drop(columns=['timestamp',], inplace=True)
    df = df.reset_index() # set date as column again
    return df

def add_forex_features(df:pd.DataFrame)->pd.DataFrame:
    forex_df = dao.get_forex()
    df = df.merge(forex_df, left_on='date', right_on='date', how='left')
    return df

def add_index_features(df:pd.DataFrame)->pd.DataFrame:
    index_df = dao.get_index()
    df = df.merge(index_df, left_on='date', right_on='date', how='left')
    return df

def add_financial_features(df:pd.DataFrame)->pd.DataFrame:
    """ Add features to the dataframe, and return it. """
    trd = util.context['target_return_days']
    df = df.copy()
    # -------------------------
    # RETURNS
    # -------------------------
    df["log_return"] = np.log(df["close"] / df["close"].shift(trd))

    # -------------------------
    # MOMENTUM
    # -------------------------
    df["mom_5"] = df["close"].pct_change(5)
    df["mom_10"] = df["close"].pct_change(10)
    df["mom_20"] = df["close"].pct_change(20)

    # -------------------------
    # VOLATILITY
    # -------------------------
    df["vol_10"] = df["log_return"].rolling(10).std()
    df["vol_20"] = df["log_return"].rolling(20).std()

    # volatility ratio
    df["vol_ratio"] = df["vol_10"] / df["vol_20"]

    # -------------------------
    # MEAN REVERSION
    # -------------------------
    ma_20 = df["close"].rolling(20).mean()
    std_20 = df["close"].rolling(20).std()

    df["zscore_20"] = (df["close"] - ma_20) / std_20

    # -------------------------
    # TREND STRENGTH
    # -------------------------
    ma_50 = df["close"].rolling(50).mean()
    df["trend_50"] = (df["close"] - ma_50) / ma_50

    # -------------------------
    # VOLUME SIGNAL
    # -------------------------
    vol_mean_20 = df["volume"].rolling(20).mean()
    vol_std_20 = df["volume"].rolling(20).std()

    df["volume_zscore"] = (df["volume"] - vol_mean_20) / vol_std_20

    # -------------------------
    # INTRADAY RANGE
    # -------------------------
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]

    # -------------------------
    # CLOSE POSITION IN RANGE
    # -------------------------
    df["close_range"] = (df["close"] - df["low"]) / (df["high"] - df["low"])

    # -------------------------
    # MOMENTUM VOL ADJUSTED
    # -------------------------
    df["mom_vol_adj"] = df["mom_20"] / df["vol_20"]

    df["target"] = df["log_return"].shift(-trd) / df["vol_20"]
    df["target"] = df["target"].clip(util.context['clip_min'], util.context['clip_max'])
    df.dropna(subset=["log_return",
                      "mom_5",
                      "mom_10",
                      "mom_20",
                      "vol_10",
                      "vol_20",
                      "vol_ratio",
                      "zscore_20",
                      "trend_50",
                      "volume_zscore", "mom_vol_adj"], inplace=True)
    return df

def cross_sectional_momentum_rank(df):
    """ Define cross-sectional momentum rank feature. """
    df = df.copy()

    # momentum 20 giorni
    df["mom_20"] = df.groupby('symbol')["close"].pct_change(20)
    # rank cross-sectional per data
    mean = df.groupby("date")["mom_20"].transform("mean")
    std = df.groupby("date")["mom_20"].transform("std")

    df["mom_rank"] = (df["mom_20"] - mean) / std

    return df

def volume_shock_feature(df):
    """ Define volume shock feature as the ratio between 
    current volume and 20-day moving average of volume. """
    df = df.copy()
    vol_ma = df.groupby('symbol')["volume"].transform(
        lambda x: x.rolling(20).mean()
    )
    df["volume_shock"] = df["volume"] / vol_ma
    return df

def volatility_regime(df):
    """ Define volatility regime feature as the ratio 
    between 10-day and 60-day rolling volatility. """
    df = df.copy()
    log_return = np.log(df["close"] / df["close"].shift(1))
    vol_10 = log_return.groupby(df['symbol']).transform(
        lambda x: x.rolling(10).std()
    )
    vol_60 = log_return.groupby(df['symbol']).transform(
        lambda x: x.rolling(60).std()
    )
    df["vol_regime"] = vol_10 / vol_60
    return df

def reorder_columns(df)->pd.DataFrame:
    """ 
    Reorder columns in the dataframe, in alfanumeric order.
    But volume column is removed, because is bad for the model.
    """
    cols:list = df.columns.tolist()
    if 'volume' in cols:
        cols.remove('volume')
    cols.sort()
    df = df[cols]
    return df

def add_country(df):
    df_model_portfolio = pd.read_csv(os.path.join(util.context['model_dir'], 'portfolio.csv'))

    list_countries = df_model_portfolio['Country'].unique().tolist()
    country_type = pd.api.types.CategoricalDtype(categories=list_countries)
    df_portfolio = pd.read_csv(os.path.join(util.context['portfolio_dir'], 'portfolio.csv'))
    df_countries = df_portfolio[['Symbol', 'Country']].sort_values(by='Country')
    df = df.merge(df_countries, left_on='symbol', right_on='Symbol', how='left')
    df['Country'] = df['Country'].astype(country_type)
    df = pd.get_dummies(df, columns=['Country'], prefix='country',dtype=int)
    df.drop(columns=['Symbol'], inplace=True)
    
    list_sectors = df_model_portfolio['Sector'].unique().tolist()
    sector_type = pd.api.types.CategoricalDtype(categories=list_sectors)
    df_sector = df_portfolio[['Symbol', 'Sector']].sort_values(by='Sector')
    df = df.merge(df_sector, left_on='symbol', right_on='Symbol', how='left')
    df['Sector'] = df['Sector'].astype(sector_type)
    df = pd.get_dummies(df, columns=['Sector'], prefix='sector',dtype=int)
    df.drop(columns=['Symbol'], inplace=True)
    
    # df_industry = df_portfolio[['Symbol', 'Industry']].sort_values(by='Industry')
    # df = df.merge(df_industry, left_on='symbol', right_on='Symbol', how='left')
    # df = pd.get_dummies(df, columns=['Industry'], prefix='industry',dtype=int)
    # df.drop(columns=['Symbol'], inplace=True)
    
    return df


def add_features_portfolio()->bool:
    """ Add features to the portfolio, and save to parquet. """
    count = len(util.context['symbols'])
    dfs = []
    for i, symbol in enumerate(util.context['symbols']):
        filename = os.path.join(util.context['hist_dir'], f"{symbol}.parquet")
        if os.path.exists(filename):
            logger.info("Processing %s (%d/%d)...", symbol, i+1, count)
            df = pd.read_parquet(filename)
            df = add_time_features(df)
            df = add_financial_features(df)
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = cross_sectional_momentum_rank(df)
    df = volume_shock_feature(df)
    df = volatility_regime(df)
    df = add_forex_features(df)
    df = add_index_features(df)
    df = add_country(df)
    df = reorder_columns(df)
    # the close and low prices are not useful for the model, so we can drop them
    df.drop(columns=['close','low','high','open'], inplace=True)
    filename = os.path.join(util.context['portfolio_dir'], "features.parquet")
    df.to_parquet(filename, index=False)
    return True

def apply_prediction()->pd.DataFrame:
    """ Apply the model to the portfolio, and save predictions to parquet. """
    import keras
    logger.info("Applying model to the portfolio...")
    filename = os.path.join(util.context['portfolio_dir'], "features.parquet")
    filename_model = os.path.join(util.context['model_dir'], "model.keras")
    filename_out = os.path.join(util.context['portfolio_dir'], "predictions.parquet")
    model = keras.saving.load_model(filename_model)
    df = pd.read_parquet(filename)
    df = df.reset_index(drop=True)
    df = df[df['date'] == df["date"].max()]
    logger.info("Applying model to %d assets... with date=%s", len(df), df['date'].max())
    df_asset = df[['symbol', 'date']]
    df = df.drop(columns=['date','symbol','target'])
    # normalize df
    mins = pd.read_csv(os.path.join(util.context['model_dir'], "mins.csv"), index_col='col')['value']
    maxs = pd.read_csv(os.path.join(util.context['model_dir'], "maxs.csv"), index_col='col')['value']
    mins = mins.drop('target', errors='ignore')
    maxs = maxs.drop('target', errors='ignore')
    df = (df - mins) / (maxs - mins)
    x = df.values
    predictions = model.predict(x, verbose=0)
    # denormalize
    label_min = util.context['label_min']
    label_max = util.context['label_max']
    predictions = predictions * (label_max - label_min) + label_min
    predictions = predictions.clip(util.context['clip_min'], util.context['clip_max'])
    df['prediction'] = predictions
    # rimetto i nomi degli asset e le date
    df['symbol'] = df_asset['symbol']
    df['date'] = df_asset['date']
    df.to_parquet(filename_out, index=False)
    logger.info("Predictions saved to parquet, row count: %d", len(df))
    return df

def define_weight():
    """ Define weights for the portfolio, based on predictions and volatility. """
    logger.info("Defining weights for the portfolio...")
    filename_in = os.path.join(util.context['portfolio_dir'], "predictions.parquet")
    filename_out = os.path.join(util.context['portfolio_dir'], "weights.parquet")
    df = pd.read_parquet(filename_in)
    df = df[['symbol','prediction','vol_20']]
    df['weight'] = df['prediction'].clip(lower=util.context['clip_min'], upper=util.context['clip_max']) / df['vol_20']
    # ordina per peso decrescente
    df = df.sort_values(by='weight', ascending=False)
    df_long = df[df['weight'] > 0].head(util.context['n_long'])
    df_short = df[df['weight'] < 0].tail(util.context['n_short'])
    df = pd.concat([df_long, df_short])
    weight_sum = df['weight'].abs().sum()
    df['weight'] = df['weight'] / weight_sum
    assert(df['weight'].abs().sum() > 0.99 and df['weight'].abs().sum() < 1.01)
    logger.info("Weights defined, row count: %d", len(df))
    df.to_parquet(filename_out, index=False)
    return df

def build_new_position():
    """
    Calculate positions, based on weights.
    columns: symbol, weight_new, qty_old
    """
    logger.info("Building new positions for the portfolio...")
    filename_in = os.path.join(util.context['portfolio_dir'], "weights.parquet")
    filename_out = os.path.join(util.context['portfolio_dir'], "position_new.parquet")
    df_new = pd.read_parquet(filename_in)
    df_new = df_new[['symbol', 'weight']]
    df_new.columns = ['symbol', 'weight_new']
    df_old = dao.get_actual_position()
    df_old = df_old[['symbol', 'qty_old']]
    df = pd.merge(df_new, df_old, on='symbol', how='outer').fillna(0)
    logger.info("New position built with %d assets", len(df))
    df.to_parquet(filename_out, index=False)

def define_new_quantity():
    """ Calculate quantity for the new position. 
    colunms: symbol, weight_new, qty_old, price, value_new, qty_new, qty_diff, qty_diff_perc
    """
    logger.info("Defining new quantities for the portfolio...")
    filename_in = os.path.join(util.context['portfolio_dir'], "position_new.parquet")
    filename_out = os.path.join(util.context['portfolio_dir'], "position_new_qty.parquet")
    df = pd.read_parquet(filename_in)
    df_price = dao.get_latest_price(df['symbol'].tolist())
    df = pd.merge(df, df_price, on='symbol', how='outer').fillna(0)
    equity = dao.get_equity()
    capital = equity * util.context['risk_percentage']
    df['value_new'] = df['weight_new'] * capital
    df['qty_new'] = (df['value_new'] / df['price']).round()
    df['qty_diff'] = df['qty_new'] - df['qty_old']
    denominator = np.where(df['qty_old'] != 0, df['qty_old'], df['qty_new'])
    df['qty_diff_perc'] = df['qty_diff'] / denominator
    # remove rows where qty_diff is less than the minimum percentage in absolute value
    df = df[df['qty_diff_perc'].abs() > util.context['qty_diff_perc_min']]
    df = df[['symbol', 'qty_old', 'qty_new', 'qty_diff', 'price', 'weight_new', 'value_new', 'qty_diff_perc']]
    df.to_parquet(filename_out, index=False)
    logger.info("New quantities defined, row count: %d", len(df))


def execution():
    """ Execute the trades to rebalance the portfolio. """
    logger.info("Executing trades for the portfolio %s", util.context['portfolio_id'])
    client = util.get_trading_client()
    filename_in = os.path.join(util.context['portfolio_dir'], "position_new_qty.parquet")
    df = pd.read_parquet(filename_in)
    df = df[['symbol', 'qty_old', 'qty_new','qty_diff']]

    for _, row in df.iterrows():
        symbol = row['symbol']
        qty_old = row['qty_old']
        qty_new = row['qty_new']
        qty_diff = row['qty_diff']
        logger.info("Processing %s: qty_old=%s, qty_new=%s, qty_diff=%s",
                    symbol, qty_old, qty_new, qty_diff)

        # "zero" handling
        if qty_new == 0:
            client.close_position(symbol)
        elif qty_old == 0 and qty_new > 0:
            trade.place_order_buy(client, symbol, qty_new)
        elif qty_old == 0 and qty_new < 0:
            trade.place_order_short(client, symbol, abs(qty_new))

        # qty old > 0 long position existing
        elif qty_old > 0:
            if qty_new > 0 and qty_diff > 0:
                trade.place_order_buy(client, symbol, qty_diff)
            elif qty_new > 0 and qty_diff < 0:
                trade.place_order_sell(client, symbol, abs(qty_diff))
            elif qty_new < 0:
                client.close_position(symbol)
                trade.place_order_short(client, symbol, abs(qty_new))

        # qty old < 0 short position existing
        elif qty_old < 0:
            if qty_new < 0 and qty_diff > 0:
                trade.place_order_buy(client, symbol, abs(qty_diff))
            elif qty_new < 0 and qty_diff < 0:
                trade.place_order_short(client, symbol, abs(qty_diff))
            elif qty_new > 0:
                client.close_position(symbol)
                trade.place_order_buy(client, symbol, qty_new)

def init_portfolio(initial_amount:float)->None:
    """
    Initialize the portfolio by creating an empty actual position file with only cash.
    """
    filename_out = os.path.join(util.context['portfolio_dir'], "actual_positions.parquet")
    df_actual = pd.DataFrame({
        'date': [date.today()],
        'symbol': [util.CASH_SYMBOL],
        'qty': 1,
        'price': initial_amount,
        'value': initial_amount
    })
    for symbol in util.context['symbols']:
        new_row ={
            'date': date.today(),
            'symbol': symbol,
            'qty': 0,
            'price': 0,
            'value': 0
        }
        df_actual = pd.concat([df_actual, pd.DataFrame([new_row])], ignore_index=True)

    df_actual.to_parquet(filename_out, index=False)
