import logging

LOG_LEVEL = logging.INFO
SUBPROCESS_LOG_LEVEL = logging.ERROR

def get_logger(name, level=LOG_LEVEL):
    logging.basicConfig(
        level=level, 
        format= '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(name)
    return logger
