import os
from datetime import date,datetime
import pandas as pd
import numpy as np
import hayai_util as util
import keras



def add_features(df:pd.DataFrame)->pd.DataFrame:
    trd = util.context['target_return_days']
    # df['timestamp'] = pd.to_datetime(df['timestamp'])

    # #features about time
    df['date'] = df['timestamp'].dt.date
    df.drop(columns=['timestamp'], inplace=True)
    # df = df.set_index('date')
    # df['day_of_week'] = df['timestamp'].dt.dayofweek
    # df['index_of_max'] = date.today()
    # for i,row in df.iterrows():
    #     df.loc[i,'index_of_max'] = df['close'][:i].idxmax()
    # df['timedelta_since_high'] = (df.index - df['index_of_max'])
    # df['time_since_high'] = df.apply(lambda x: x['timedelta_since_high'].days, axis=1)
    # df.drop(columns=['index_of_max','timedelta_since_high'], inplace=True)

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

    df = df.copy()

    # momentum 20 giorni
    df["mom_20"] = df.groupby('symbol')["close"].pct_change(20)
    # rank cross-sectional per data
    mean = df.groupby("date")["mom_20"].transform("mean")
    std = df.groupby("date")["mom_20"].transform("std")

    df["mom_rank"] = (df["mom_20"] - mean) / std

    return df

def volume_shock_feature(df):

    df = df.copy()

    vol_ma = df.groupby('symbol')["volume"].transform(
        lambda x: x.rolling(20).mean()
    )

    df["volume_shock"] = df["volume"] / vol_ma

    return df

def volatility_regime(df):

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

def add_features_portfolio()->bool:
    count = len(util.context['symbols'])
    dfs = []
    for i, symbol in enumerate(util.context['symbols']):
        filename = os.path.join(util.context['hist_dir'], f"{symbol}.parquet")
        if os.path.exists(filename):
            print(f"Processing {symbol} ({i+1}/{count})...")
            df = pd.read_parquet(filename)
            df = add_features(df)
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = cross_sectional_momentum_rank(df)
    df = volume_shock_feature(df)
    df = volatility_regime(df)
    filename = os.path.join(util.context['portfolio_dir'], "features.parquet")
    df = df[['symbol','date','log_return','mom_5','mom_10','mom_rank','vol_10',
                'vol_20','vol_ratio','zscore_20','trend_50','volume_zscore',
                'volume_shock','vol_regime','target']]
    df.to_parquet(filename, index=False)
    return True

def apply_prediction()->pd.DataFrame:
    filename = os.path.join(util.context['portfolio_dir'], "features.parquet")
    filename_model = os.path.join(util.context['portfolio_dir'], "model.keras")
    model = keras.saving.load_model(filename_model)
    df = pd.read_parquet(filename)
    df = df.reset_index(drop=True)
    df = df[df['date'] == df["date"].max()]
    print(f"Applying model to {len(df)} assets... with date={df['date'].max()}")
    df_asset = df[['symbol', 'date']]
    df = df.drop(columns=['date','symbol','target'])
    x = df.values
    predictions = model.predict(x, verbose=0)
    df['prediction'] = predictions
    #denormalize
    label_min = util.context['label_min']
    label_max = util.context['label_max']
    df['prediction'] = df['prediction'] * (label_max - label_min) + label_min
    df['weight'] = df['prediction'].clip(lower=-5, upper=5) / df['vol_20']
    # rimetto i nomi degli asset e le date
    df['symbol'] = df_asset['symbol']
    df['date'] = df_asset['date']
    filename_out = os.path.join(util.context['portfolio_dir'], "predictions.parquet")
    df.to_parquet(filename_out, index=False)
    return df

def add_price(df:pd.DataFrame)->pd.DataFrame:
    df = df.copy()
    df['close'] = 0.0
    for asset in df['symbol']:
        filename = os.path.join(util.context['hist_dir'], f"{asset}.parquet")
        df_price = pd.read_parquet(filename)
        df_price = df_price[df_price['timestamp'] == df_price['timestamp'].max()]
        if not df_price.empty:
            price = df_price['close'].values[0]
            df.loc[df['symbol'] == asset, 'close'] = price
        else:
            print(f"Warning: No price data for {asset} on {df['date'].max()}")
    return df

def positions_construction():
    """ Calculate positions, percentage relative to the total portfolio, based on the predictions and the volatility of each asset. """
    filename = os.path.join(util.context['portfolio_dir'], "predictions.parquet")
    df = pd.read_parquet(filename)
    df = df.sort_values(by='weight', ascending=False)
    df_long = df[df['weight'] > 0].head(util.context['n_long'])
    df_long['side'] = 'long'
    df_short = df[df['weight'] < 0].tail(util.context['n_short'])
    df_short['side'] = 'short'
    df = pd.concat([df_long, df_short])
    sum_weights = df['weight'].abs().sum()
    df['position_perc'] = df['weight'].abs() / sum_weights
    print("position total sum:", df['position_perc'].sum())
    assert(df['position_perc'].sum() > 0.99 and df['position_perc'].sum() < 1.01)
    filename_out = os.path.join(util.context['portfolio_dir'], "positions.parquet")
    df[['symbol','date','side','position_perc']].to_parquet(filename_out, index=False)
    return df


    