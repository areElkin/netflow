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

def _gen_logger() -> "logging.Logger":
    from rich.console import Console
    from rich.logging import RichHandler
    # RichHandler.KEYWORDS.append('MSG')

    logger = logging.getLogger(__name__)
    # logger = logging.getLogger(name)
    # colorlog.install(logger=logger)

    # .Note:: Comment this line to show warnings
    logging.captureWarnings(True)  


    logger.setLevel(logging.INFO) # logging.INFO)
    # console = Console(force_terminal=True)
    console = Console()
    # if console.is_jupyter is True:
    #     console.is_jupyter = False
    ch = RichHandler(console=console, show_path=False,
                     show_time=False,
                     keywords=RichHandler.KEYWORDS + ['MSG'],
                     )


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

logger = _gen_logger()
