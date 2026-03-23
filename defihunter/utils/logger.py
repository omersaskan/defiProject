import logging
import os
import sys

def setup_logger(name="defihunter", level=logging.INFO):
    """
    Setup a centralized logger with console and file handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

# Default instance
logger = setup_logger()
