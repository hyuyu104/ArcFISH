import sys
import logging

logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO, 
    format="%(message)s"
)

from . import utils as tl
from . import plot as pl

from .analysis import loop
from .analysis import domain
from .analysis import simulate

sys.modules.update({
    f"{__name__}.{m}": globals()[m]
    for m in ["pl", "tl", "loop", "domain"]
})