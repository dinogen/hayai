""" main """
import argparse
import hayai_util as util
import hayai_dao
import hayai_bo
import hayai_trade

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Run the trading application.')
    args.add_argument('--portfolio_id', type=str, help='ID of the portfolio to use',required=True)
    args.add_argument('--train-dataset', help='Build the dataset for training the model',action='store_true')
    args.add_argument('--signal-dataset', help='Build the dataset for prediction',action='store_true')
    args.add_argument('--execute-trades', help='Execute trades based on the latest predictions',action='store_true')
    portfolio_id = args.parse_args().portfolio_id
    train_dataset = args.parse_args().train_dataset
    signal_dataset = args.parse_args().signal_dataset
    execute_trades = args.parse_args().execute_trades
    context = util.create_context(portfolio_id)
    if train_dataset:
        hayai_dao.fetch_data_portfolio(365*5)
        hayai_bo.add_features_portfolio()
    if signal_dataset:
        hayai_dao.fetch_data_portfolio(365*5)
        hayai_bo.add_features_portfolio()
        hayai_bo.apply_prediction()
        hayai_bo.positions_construction()
    if execute_trades:
        hayai_trade.execution()