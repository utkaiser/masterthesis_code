import logging
import sys
sys.path.append("..")
from models.model_utils import setup_logger

if __name__ == '__main__':

    logging.basicConfig(filemode='a',
                        format='%(asctime)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    logger1 = logging.getLogger('logger1')
    handler1 = logging.FileHandler('log1.log')
    logger1.addHandler(handler1)

    logger2 = logging.getLogger('logger2')
    handler2 = logging.FileHandler('log2.log')
    logger2.addHandler(handler2)

    logger1.info("a")
    logger2.info("b")






