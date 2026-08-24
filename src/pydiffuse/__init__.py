from email.utils import parseaddr
from importlib.metadata import metadata, version

__version__ = version("pydiffuse")
__author__ = parseaddr(metadata("pydiffuse")["Author-email"])[0]
