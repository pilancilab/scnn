"""Utilities for interfacing between public API and
:module:`scnn.private`."""

import logging
import lab


def get_logger(
    name: str, verbose: bool = False, debug: bool = False, log_file: str = None
) -> logging.Logger:
    """Construct a logging.Logger instance with an appropriate configuration.

    Args:
        name: name for the Logger instance.
        verbose: (optional) whether or not the logger should print verbosely (ie. at the INFO level).
            Defaults to False.
        debug: (optional) whether or not the logger should print in debug mode (ie. at the DEBUG level).
            Defaults to False.
        log_file: (optional) path to a file where the log should be stored. The log is printed to stdout when 'None'.

    Returns:
         Instance of logging.Logger.
    """

    level = logging.WARNING
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO

    logging.basicConfig(level=level, filename=log_file)
    logger = logging.getLogger(name)
    logging.root.setLevel(level)
    logger.setLevel(level)
    return logger


def set_device(device: str = "cpu", dtype="float32", seed: int = 778):
    """Set the device to be used by LAB, the global dtype, and random seeds."""
    if device == "cpu":
        lab.set_backend("numpy")
        lab.set_device(device)
    elif device == "cuda":
        lab.set_backend("torch")
        lab.set_device(device)

    lab.set_dtype(dtype)
    lab.set_seeds(seed)
