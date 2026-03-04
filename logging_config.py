"""
Add at every source file
import logging_config
logger = logging_config.create_logger(__name__)

"""
import logging
from  datetime import date

def create_logger(nome_logger:str):
    logger = logging.getLogger(nome_logger)
    logger.setLevel(logging.DEBUG)
    oggi = str(date.today())
    nomefile = f"hayai.{oggi}.log"
    ch = logging.FileHandler(nomefile)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger


