import logging
import os

def setup_logger(log_dir, log_filename="experiment.log", level=logging.INFO):
    """
    Sets up a logger to log to both console and a file.

    Args:
        log_dir (str): Directory where the log file will be saved.
        log_filename (str, optional): Name of the log file. Defaults to "experiment.log".
        level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('my_project_logger')
    logger.setLevel(level)

    # Formatter for all handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler logs everything to file
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Stream handler prints to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    # Remove any existing handlers (important if called multiple times)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
