import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide,TimeInForce


portfolio_id = 'medium_tech_usa'
initial_amount = 100000

api_key = 'PKEAC5FYDZJSGJDRQT3JARF3KN'
secret_key = 'DZfXDQPViAbrZNsK7iETuqWV6fFGQfvb55TkzkBEeJV8'
client1 = TradingClient(api_key, secret_key)

api_key = 'PKWCB7TDKCHZWTU6NBFWYX6WQW'
secret_key = '4nTFfMLb6PUHVer72yhwPcifJ51WpsKy8M8zBMd4B2ZC'
client_control = TradingClient(api_key, secret_key)

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
        client_control.submit_order(market_order_data)
        print(f"Buying {amount} of {symbol}")
    except Exception as e:
        print(f"Error buying {symbol}: {e}")