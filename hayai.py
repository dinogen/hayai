""" main """
import argparse
import hayai_util as util
import hayai_dao
import hayai_bo
import hayai_msg as msg
import hayai_log
logger = hayai_log.create_logger(__name__)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Run the trading application.')
    args.add_argument('-p','--portfolio-id', type=str, help='ID of the portfolio to use',required=True)
    args.add_argument('-i','--ingestion', help='Build an updated dataset for training the model',action='store_true')
    args.add_argument('-s','--build-signals', help='Calculate signals and weights',action='store_true')
    args.add_argument('-n','--new-position', help='Calculate the new position of portfolio',action='store_true')
    args.add_argument('-e','--execute-trades', help='Execute trades based on the new position',action='store_true')
    portfolio_id = args.parse_args().portfolio_id
    ingestion = args.parse_args().ingestion
    build_signals = args.parse_args().build_signals
    new_position = args.parse_args().new_position
    execute_trades = args.parse_args().execute_trades
    logger.info("Starting HAYAI with portfolio_id=%s, ingestion=%s, build_signals=%s, new_position=%s, execute_trades=%s", 
                portfolio_id, ingestion, build_signals, new_position, execute_trades)
    context = util.create_context(portfolio_id)
    if ingestion:
        logger.info("Starting data ingestion...")
        hayai_dao.fetch_quotes_portfolio(365*5)
        hayai_bo.add_features_portfolio()
        logger.info("Data ingestion finished.")
    if build_signals:
        logger.info("Calculating signals and weights...")
        hayai_bo.apply_prediction()
        hayai_bo.define_weight()
        logger.info("Signals and weights calculation finished.")
        msg.send_file(hayai_log.log_filename(), caption='HAYAI log file after building signals and weights.')
    if new_position:
        logger.info("Calculating new position...")
        hayai_bo.build_new_position()
        hayai_bo.define_new_quantity()
        logger.info("New position calculation finished.")
    if execute_trades:
        logger.info("Executing trades...")
        hayai_bo.execution()
        logger.info("Trades execution finished.")
        msg.send_message("HAYAI has executed trades for portfolio '%s'. Go check on Alpaca" % portfolio_id)
    logger.info("HAYAI finished.")