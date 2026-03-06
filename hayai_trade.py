import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
import hayai_util as util

import logging_config
logger = logging_config.create_logger(__name__)

def place_order_buy(tc:TradingClient,symbol:str,qty:float):
    assert qty > 0, "Quantity must be positive for buy orders."
    logger.info(f"Placing buy order for {symbol} with quantity {qty}")
    market_order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY)

    # Market order
    try: 
        market_order = tc.submit_order(order_data=market_order_data)
    except APIError as e:
        logger.error(f"Error placing buy order for {symbol}: {e}")

def place_order_sell(tc:TradingClient,symbol:str,qty:float):
    assert qty > 0, "Quantity must be positive for sell orders."
    qty = round(qty)
    logger.info(f"Placing sell order for {symbol} with quantity {qty}")
    if qty < 1:
        print(f"Quantity {qty} is less than 1, skipping order for {symbol}.")
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
        logger.error(f"Error placing sell order for {symbol}: {e}")

def place_order_short(tc:TradingClient,symbol:str,qty:float):
    assert qty > 0, "Quantity must be positive for short orders."
    qty = round(qty)
    logger.info(f"Placing short order for {symbol} with quantity {qty}")
    market_order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY)

    # Market order
    try:
        market_order = tc.submit_order(order_data=market_order_data)
    except APIError as e:
        logger.error(f"Error placing short order for {symbol}: {e}")

