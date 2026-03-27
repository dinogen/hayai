import os
from datetime import date
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
import yfinance as yf
import hayai_util as util
import hayai_dao as dao

import hayai_log
logger = hayai_log.create_logger(__name__)

def place_order_buy(symbol:str,qty:float,tc:TradingClient=None):
    assert qty > 0, "Quantity must be positive for buy orders."
    logger.info("Placing buy order for %s with quantity %s", symbol, qty)
    if util.context['data_source'] == 'alpaca':
        _place_order_buy_alpaca(tc, symbol, qty)
    if util.context['data_source'] == 'yfinance':
        _place_order_buy_yfinance(symbol, qty)

def _place_order_buy_alpaca(tc:TradingClient,symbol:str,qty:float):
    market_order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY)
    try: 
        market_order = tc.submit_order(order_data=market_order_data)
    except APIError as e:
        logger.error("Error placing buy order for %s: %s", symbol, e)

def _place_order_buy_yfinance(symbol:str,qty:float):
    """
    Modify the actual position file.
    Implement a buy order by increasing the quantity of the symbol in the actual position file
    and decreasing cash.
    Save the new actual position as (symbol, qty, price, value)
    """
    df_new_position = pd.read_parquet(os.path.join(util.context['portfolio_dir'], "position_new_qty.parquet"))
    df_actual = dao.get_actual_position_yfinance()
    filename_out = os.path.join(util.context['portfolio_dir'], "actual_positions.parquet")
    cash = df_actual[df_actual['symbol'] == util.CASH_SYMBOL]['qty'].iloc[0] if util.CASH_SYMBOL in df_actual['symbol'].values else 0
    if symbol in df_actual['symbol'].values:
        price = df_new_position[df_new_position['symbol'] == symbol]['price'].iloc[0]
        value = price * qty
        cash -= value
        df_actual.loc[df_actual['symbol'] == symbol, 'qty'] += qty
        df_actual.loc[df_actual['symbol'] == symbol, 'value'] += value
        df_actual.loc[df_actual['symbol'] == symbol, 'price'] = price
        df_actual.loc[df_actual['symbol'] == util.CASH_SYMBOL, 'qty'] = cash
    else:
        pass
    df_actual.to_parquet(filename_out, index=False)

def place_order_sell(tc:TradingClient,symbol:str,qty:float):
    assert qty > 0, "Quantity must be positive for sell orders."
    qty = round(qty)
    logger.info("Placing sell order for %s with quantity %s", symbol, qty)
    if qty < 1:
        logger.warning("Quantity %s is less than 1, skipping order for %s.", qty, symbol)
        return
    market_order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY)
    # Market order
    try:
        market_order = tc.submit_order(order_data=market_order_data)
    except APIError as e:
        logger.error("Error placing sell order for %s: %s", symbol, e)

def place_order_short(tc:TradingClient,symbol:str,qty:float):
    assert qty > 0, "Quantity must be positive for short orders."
    qty = round(qty)
    logger.info("Placing short order for %s with quantity %s", symbol, qty)
    market_order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY)

    # Market order
    try:
        market_order = tc.submit_order(order_data=market_order_data)
    except APIError as e:
        logger.error("Error placing short order for %s: %s", symbol, e)

