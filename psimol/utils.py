import os 
import sys
import logging 
import argparse
import numpy as np
from ruamel.yaml import YAML
from typing import Any, Dict, List, Literal, Tuple, Union


class CustomParser(argparse.ArgumentParser):
    """Custom parser class."""
    
    def error(self, message):
        """Print help with error message."""
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


class CustomLoggingFormatter(logging.Formatter):
    """Custom logging formatter."""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[36;20m"
    reset = "\x1b[0m"
    format = "[%(asctime)s] [%(levelname)s] %(message)s"

    FORMATS = {
        logging.DEBUG: blue + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }
    
    def format(self, record):
        """Format the log record."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    

def get_package_directory() -> str:
    """Helper function to get the directory of the package.

    Returns:
        str: Path to the package directory.
    """
    return os.path.dirname(os.path.abspath(__file__))


def setup_logging(loglevel: int) -> None:
    """Setup logging for the package.
    
    Args:
        loglevel (int): Log level to set.
    """
    custom_logging = logging.StreamHandler()
    custom_logging.setLevel(loglevel)
    custom_logging.setFormatter(CustomLoggingFormatter())

    logging.basicConfig(
        level=loglevel,
        handlers=[custom_logging],
        force=True    
    )


def get_atom_config(symbol: str) -> Dict[str, Any]:
    """Get the configuration of an atom.

    Args:
        symbol (str): Atom symbol.

    Returns:
        Dict[str, Any]: Configuration of an atom.
    """
    package_path = get_package_directory()
    atom_config_path = os.path.join(package_path, 'configs', 'atom_properties.yml')

    if not os.path.exists(atom_config_path):
        logging.fatal(f'Atom configuration file not found at {atom_config_path}.')
        sys.exit(1)

    yaml = YAML(typ='safe')
    with open(atom_config_path, 'r') as file:
        atom_config = yaml.load(file)

    if symbol not in atom_config:
        logging.error(f'Atom configuration not found for provided symbol {symbol}.')
        sys.exit(1)
    
    return atom_config[symbol]

def euclidean_distance(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Calculate the Euclidean distance between two vectors.

    Args:
        coords1 (np.ndarray): First vector coordinates.
        coords2 (np.ndarray): Second vector coordinates.

    Returns:
        float: Euclidean distance between vectors.
    """
    return np.linalg.norm(coords1 - coords2)
