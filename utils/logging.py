import sys
import logging
from typing import Optional
import os


def set_default_logger(experiment_path: Optional[str], debug=False):
    log_format = '%(asctime)s|%(levelname)-8s| %(message)s'

    formatter = logging.Formatter(log_format)

    handlers = []
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    if experiment_path is not None:
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        filename = os.path.join(experiment_path, 'experiment.log')
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        handlers=handlers,
        level=level
    )
