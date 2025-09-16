import logging
import sys
from typing import Optional


class Logger:
    """
    A simple logger class that wraps around Python's built-in logging module.
    Provides methods for logging messages at different severity levels.
    """
    def __init__(self, name: str = "MAC-KG", level=logging.INFO, log_to_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # 避免重复添加 handler

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Optional file handler
        if log_to_file:
            file_handler = logging.FileHandler(log_to_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

    def exception(self, message):
        """
        Logs an exception with traceback.
        Must be called inside an exception handler.
        """
        self.logger.exception(message)
