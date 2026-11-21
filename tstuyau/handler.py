import logging
import time
from contextlib import contextmanager

## adding custom level (between info and debug for normal info + costs info (time & memory))
COSTS_LEVEL = 15
logging.addLevelName(COSTS_LEVEL, "COSTS")

setattr(logging, "COSTS", COSTS_LEVEL)

def log_costs(self, message, *args, **kws):
    if self.isEnabledFor(COSTS_LEVEL):
        self._log(COSTS_LEVEL, message, args, **kws)

logging.Logger.costs = log_costs


logger = logging.getLogger(__name__)

_FORMAT = '%(asctime)s:%(levelname)s:%(lineno)s:%(module)s.%(funcName)s:%(message)s'

_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(_FORMAT, '%H:%M:%S'))
logger.addHandler(_handler)


# Avoid duplicate lines from the root logger
logger.propagate = False

# Default until the config is loaded
logger.setLevel(logging.INFO)

def set_log_level(level):
    """Set the package logger level from a string like 'debug' or 'INFO'."""
    level_num = getattr(logging, str(level).upper(), None)
    if not isinstance(level_num, int):
        logger.warning(f'Unknown log_level {level!r}; keeping current level')
        return
    logger.setLevel(level_num)
    ## to get other libraries' log level:
    #logging.basicConfig(level=level_num)

@contextmanager
def timed(label):
    t0 = time.perf_counter()
    yield
    #current_logger = logging.getLogger(label) 
    #if current_logger.isEnabledFor(COSTS_LEVEL):
    logger.costs(f"[time] {label}: {time.perf_counter() - t0:.1f}s")
