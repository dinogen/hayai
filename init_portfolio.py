import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide,TimeInForce


portfolio_id = 'mix_2'
initial_amount = 100000

api_key = ''
secret_key = ''
client1 = TradingClient(api_key, secret_key)



df = pd.read_csv(f"data\\{portfolio_id}\\portfolio.csv")
amount = int(initial_amount / len(df))
for index, row in df.iterrows():
    try:
        symbol = row['Symbol']
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            notional=amount,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        client1.submit_order(market_order_data)
        print(f"Buying {amount} of {symbol}")
    except Exception as e:
        print(f"Error buying {symbol}: {e}")