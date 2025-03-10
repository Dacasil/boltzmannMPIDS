"""Initialize the package."""

from importlib import metadata as importlib_metadata

from .hello_world import hello_world
from .model import BoltzmannMachine
from .training import TrainBatch
def _get_version():
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


# Not really necessary https://gitlab.com/python-devs/importlib_metadata/-/merge_requests/125,
# but most packages do it:
__version__ = _get_version()

# __author__ = "Max Mustermann"
# __email__ = "max.mustermann@ds.mpg.de"
