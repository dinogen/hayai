import os
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from hayai_dao import get_actual_position
import hayai_util as util
import hayai_bo


def add_position_size(df:pd.DataFrame,client:TradingClient)->pd.DataFrame:
    # calculate the part of capital that is going to invest
    rp = util.context['risk_percentage']
    account = client.get_account()
    portfolio_value = float(account.portfolio_value)
    # calculate the position size in USD for each asset
    df['position_size'] = (rp * portfolio_value) * df['position_perc']
    return df

def place_order_buy(tc:TradingClient,symbol:str,qty:float):
    assert qty > 0, "Quantity must be positive for buy orders."
    if qty < 1:
        print(f"Quantity {qty} is less than 1, skipping order for {symbol}.")
        return
    market_order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY)

    # Market order
    market_order = tc.submit_order(order_data=market_order_data)

def place_order_sell(tc:TradingClient,symbol:str,qty:float):
    assert qty > 0, "Quantity must be positive for sell orders."
    if qty < 1:
        print(f"Quantity {qty} is less than 1, skipping order for {symbol}.")
        return
    market_order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY)

    # Market order
    market_order = tc.submit_order(order_data=market_order_data)

def place_order_short(tc:TradingClient,symbol:str,qty:float):
    assert qty > 0, "Quantity must be positive for short orders."
    qty = int(qty)
    if qty <= 1:
        print(f"Quantity {qty} is less than 1, skipping order for {symbol}.")
        return
    market_order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY)

    # Market order
    market_order = tc.submit_order(order_data=market_order_data)


def get_actual(client:TradingClient)->pd.DataFrame:
    # Get the actual
    portfolio = client.get_all_positions()
    df = pd.DataFrame([position.__dict__ for position in portfolio])
    df['qty'] = df['qty'].astype(float)
    df = df[['symbol', 'qty', 'side']]
    df.columns = ['symbol', 'qty_old', 'side_old']
    return df

def build_to_be(client:TradingClient)->pd.DataFrame:
    """ Set the to be"""
    filename_in = os.path.join(util.context['portfolio_dir'], "positions.parquet")
    df = pd.read_parquet(filename_in)
    df = hayai_bo.add_price(df)
    df = add_position_size(df, client)
    df['qty_new'] = df['position_size'] / df['close']
    df.loc[df['side'] == 'short', 'qty_new'] = df['qty_new'] * -1
    return df

def execution():
    apikey = util.context['api_key']
    secret_key = util.context['secret_key']
    client = TradingClient(api_key=apikey,secret_key=secret_key,paper=True)
    df_actual = get_actual(client)
    df_to_be = build_to_be(client)
    df = pd.merge(df_to_be, df_actual, on='symbol', how='outer')
    df = df[['symbol', 'qty_old', 'qty_new']].fillna(0)
    df['qty_diff'] = df['qty_new'] - df['qty_old']

    for _, row in df.iterrows():
        symbol = row['symbol']
        if row['qty_new'] == 0:
            client.close_position(symbol)

        elif row['qty_old'] > 0 and row['qty_new'] > 0:
            if row['qty_new'] > row['qty_old']:
                place_order_buy(client, symbol, row['qty_diff'])
            else:
                place_order_sell(client, symbol, -row['qty_diff'])

        elif row['qty_old'] < 0 and row['qty_new'] < 0:
            if row['qty_new'] < row['qty_old']:
                place_order_short(client, symbol, -row['qty_diff'])
            else:
                place_order_buy(client, symbol, -row['qty_diff'])

        elif row['qty_old'] >= 0 and row['qty_new'] < 0:
            if row['qty_old'] > 0:
                client.close_position(symbol)
            place_order_short(client, symbol, -row['qty_new'])

        elif row['qty_old'] <= 0 and row['qty_new'] > 0:
            if row['qty_old'] < 0:
                client.close_position(symbol)
            place_order_buy(client, symbol, row['qty_new'])
