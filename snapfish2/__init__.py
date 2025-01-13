import sys
import logging

logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO, 
    format="%(message)s"
)

from .utils.load import MulFish
from . import utils
from . import plot as pl

from .analysis import loop
from .analysis import domain