import logging

# package-global logger
logger = logging.getLogger('keras')

# set up basic logger to behave like print()
# this will be done on first import of a Keras module, but only if there 
# is no existing logger configuration
# to use a different configuration; first clear the handlers set here using 
#    logging.getLogger('').handlers = []
#    logging.basicConfig(...)
if not logging.getLogger('').handlers:
    logging.basicConfig(format='%(message)s', level=logging.INFO)

# make warnings issued with the warnings module be handled by logging infrastructure
logging.captureWarnings(True)
