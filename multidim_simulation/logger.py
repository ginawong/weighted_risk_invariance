import io
import sys
from tqdm import tqdm
from typing import Union
from os import PathLike
from pathlib import Path
import logging


def tqdm_logger(*args, **kwargs):
    """ Call this like tqdm.tqdm but will log progress bars properly """
    return tqdm(*args, file=_logger_as_file, **kwargs)


def init_logging(log_dir: Union[str, PathLike], prefix: str):
    log_dir = Path(log_dir).absolute()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "log.txt"
    try:
        log_path.unlink()
    except FileNotFoundError:
        pass

    handler_file = logging.FileHandler(log_path)
    handler_stream = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(f"{prefix} %(asctime)s: %(message)s")
    handler_file.setFormatter(formatter)
    handler_stream.setFormatter(formatter)

    global_logger = logging.getLogger()
    global_logger.handlers.clear()
    global_logger.setLevel(logging.INFO)
    global_logger.addHandler(handler_file)
    global_logger.addHandler(handler_stream)
    _logger_as_file.logger = global_logger


class _LoggerAsFile(io.StringIO):
    """ https://stackoverflow.com/a/41224909/2790047 """
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):
        super(_LoggerAsFile, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        if self.logger:
            self.logger.log(self.level, self.buf)


_logger_as_file = _LoggerAsFile(None)
