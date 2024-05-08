import colorlog
from functools import partial, partialmethod
import logging
import warnings
import sys

# modified from MOSCOT: https://github.com/theislab/moscot/blob/main/src/moscot/_logging.py

__all__ = ["logger"]

# Note: add custom level to hide future warnings.
# set verbose level to INFO, if want to check for other possible warnings
# and uncomment line `logging.captureWarnings(True)` to see warnings

CUSTOM_LEVEL = logging.WARNING + 5
logging.MSG = CUSTOM_LEVEL
logging.addLevelName(logging.MSG, 'MSG')
logging.Logger.msg = partialmethod(logging.Logger.log, logging.MSG)
logging.msg = partial(logging.log, logging.MSG)

logging.TRACE = logging.DEBUG + 5
logging.addLevelName(logging.TRACE, 'TRACE')
logging.Logger.trace = partialmethod(logging.Logger.log, logging.TRACE)
logging.trace = partial(logging.log, logging.TRACE)

def _gen_logger(name='') -> "logging.Logger":
    from rich.console import Console
    from rich.logging import RichHandler
    # RichHandler.KEYWORDS.append('MSG')

    # logger = logging.getLogger(__name__)
    logger = logging.getLogger(name) ########
    # logger = logging.getLogger(name)
    # colorlog.install(logger=logger)

    # .Note:: Comment this line to show warnings
    logging.captureWarnings(True)  


    # logger.setLevel(logging.INFO) # logging.INFO)
    logger.setLevel(logging.WARN) # logging.INFO)
    # console = Console(force_terminal=True)
    console = Console()
    # if console.is_jupyter is True:
    #     console.is_jupyter = False
    ch = RichHandler(console=console,
                     show_level=False,
                     show_path=False,
                     show_time=False,
                     keywords=RichHandler.KEYWORDS + ['TRACE', 'MSG'],
                     )

    # log_formatter = logging.Formatter(fmt="%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(funcName)s:%(lineno)s | %(process)d >>> %(message)s")
    log_formatter = logging.Formatter(fmt="%(name)s: %(asctime)s | %(levelname)s | %(module)s:%(funcName)s:%(lineno)s | >>> %(message)s",
                                      datefmt='%m/%d/%Y %I:%M:%S  %p')
    ch.setFormatter(log_formatter)

    
    # ch = colorlog.StreamHandler(stream=sys.stdout)
    # ch.setLevel(logging.INFO)

    # fmt = colorlog.ColoredFormatter(
    #     fmt="%(name)s: %(white)s%(asctime)s%(reset)s | %(log_color)s%(levelname)s%(reset)s | %(blue)s%(filename)s:%(yellow)s%(funcName)s:%(lineno)s%(reset)s | %(process)d >>> %(log_color)s%(message)s%(reset)s",
    #     datefmt='%m/%d/%Y %I:%M:%S  %p',
    # )
    # ch.setFormatter(fmt)
    logger.addHandler(ch)

    # this prevents double outputs
    logger.propagate = False
    return logger

# logger = _gen_logger(name=__name__)
logger = _gen_logger(name='netflow')

def set_verbose(logger, verbose="ERROR"): # "ERROR"):
    """ Set logger verbosity.

    Parameters
    ----------
    verbose: {"DEBUG", "TRACE", "INFO", "WARN", "MSG", "ERROR"}
        Verbose level. (Default = "ERROR")

        Options :
    
            - "DEBUG": show all output logs.
            - "TRACE": show detailed process logs.
            - "INFO": show only process logs to confirm things are working as expected.            
            - "WARN": show unexpected behavior, potential problem, critical message, or error logs.
            - "MSG": show critical message and error logs.
            - "ERROR": only show log if error happened.
    """
    if verbose == "DEBUG":
        # logger.setLevel(logging.DEBUG)
        level = logging.DEBUG
    elif verbose == "TRACE":
        # logger.setLevel(logging.TRACE)
        level = logging.TRACE
    elif verbose == "INFO":
        # logger.setLevel(logging.INFO)
        level = logging.INFO
    elif verbose == "WARN":
        # logger.setLevel(logging.WARNING)
        level = logging.WARNING
    elif verbose == "MSG":
        # logger.setLevel(logging.MSG)
        level = logging.MSG
    elif verbose == "ERROR":
        # logger.setLevel(logging.ERROR)
        level = logging.ERROR
    else:
        logger.error("Unrecognized verbose level, options: ['DEBUG', 'TRACE', 'INFO','WARN', 'MSG', 'ERROR'], use 'ERROR' instead")
        # logger.setLevel(logging.ERROR)
        level = logging.ERROR

    for handler in logger.handlers:
        # to check the type of the handler to target a specific outputter, e.g., uncomment the if statement:
        # if isinstance(handler, type(logging.StreamHandler())):
        if handler.level != level:
            handler.setLevel(level)
            logger.msg(f"Logging verbosity set to {level}.")





        

    
