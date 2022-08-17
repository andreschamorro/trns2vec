import os
import sys
import logging

def get_logger(module_name, log_file):
    
    # create logger with 'module_name'
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    
    # create file handler which logs even debug messages
    os.makedirs(f'{os.path.dirname(log_file)}', exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    
    # create formatter and add it to the handlers
    # formatter = logging.Formatter('[%(asctime)s] - [%(name)s] - [%(levelname)s] : %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
    verbose_formatter = logging.Formatter("[%(asctime)s.%(msecs)03d - %(levelname)s - %(process)d:%(thread)d - %(filename)s - %(funcName)s:%(lineno)d] %(message)s", datefmt="%d-%m-%Y %H:%M:%S")

    fh.setFormatter(verbose_formatter)
    ch.setFormatter(verbose_formatter)
    
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger
