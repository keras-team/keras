import logging
import sys

# package-global logger
logger = logging.getLogger('keras')

# set up logger to behave like print() by default (without client having to do anything)
if not logger.handlers:
    hdlr = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(message)s', None)
    hdlr.setFormatter(fmt)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

# make warnings issued with the warnings module be handled by logging infrastructure
logging.captureWarnings(True)
