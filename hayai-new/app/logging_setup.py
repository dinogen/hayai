import logging
import sys
from app.config import settings

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    log_file = settings.LOGS_DIR / "hayai.log"
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
