import logging
import colorlog


def get_logger():
    """
    Sets up the logger to output logs to the console.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'blue',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )
    handler.setFormatter(formatter)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.addHandler(handler)
    return logger
