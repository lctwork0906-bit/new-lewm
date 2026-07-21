import logging
import sys


def build_logger(name="aerial_2"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = build_logger()
