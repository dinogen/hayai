""" main """
import argparse
import hayai_util as util
import hayai_dao
import hayai_bo

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Run the trading application.')
    args.add_argument('--portfolio-id', type=str, help='ID of the portfolio to use',required=True)
    args.add_argument('--ingestion', help='Build an updated dataset for training the model',action='store_true')
    args.add_argument('--build-signals', help='Calculate signals and weights',action='store_true')
    args.add_argument('--new-position', help='Calculate the new position of portfolio',action='store_true')
    args.add_argument('--execute-trades', help='Execute trades based on the new position',action='store_true')
    portfolio_id = args.parse_args().portfolio_id
    ingestion = args.parse_args().ingestion
    build_signals = args.parse_args().build_signals
    new_position = args.parse_args().new_position
    execute_trades = args.parse_args().execute_trades
    context = util.create_context(portfolio_id)
    if ingestion:
        hayai_dao.fetch_quotes_portfolio(365*5)
        hayai_bo.add_features_portfolio()
    if build_signals:
        hayai_bo.apply_prediction()
        hayai_bo.define_weight()
    if new_position:
        hayai_bo.build_new_position()
        hayai_bo.define_new_quantity()
    if execute_trades:
        hayai_bo.execution()
