import logging
import yaml
from pathlib import Path


logger = logging.getLogger(__name__)

_FORMAT = '%(asctime)s:%(levelname)s:%(lineno)s:%(module)s.%(funcName)s:%(message)s'

_handler = logging.StreamHandler()
_formatter = logging.Formatter(_FORMAT, '%H:%M:%S')
_handler.setFormatter(_formatter)

logger.addHandler(_handler)
#logger.setLevel(logging.INFO)

config_file = Path(__file__).resolve().parent/'config'/'config.yaml'
with open(config_file, 'r') as f:
    tempconfig = yaml.safe_load(f)
    lev = tempconfig['log_level']
logging.basicConfig(level=getattr(logging, lev.upper()))
logger.setLevel(lev.upper())

