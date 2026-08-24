from email.utils import parseaddr
from importlib.metadata import metadata, version

__version__ = version("py-diffuser")
__author__ = parseaddr(metadata("py-diffuser")["Author-email"])[0]
