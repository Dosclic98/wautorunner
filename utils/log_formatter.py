import logging

class ColoredFormatter(logging.Formatter):
    """
    Custom logging formatter to add colors to log messages.
    """
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    formatHeader = "%(asctime)s - %(name)s - %(levelname)s"
    formatMessage = " - %(message)s"

    FORMATS = {
        logging.DEBUG: yellow + formatHeader + reset + formatMessage,
        logging.INFO: grey + formatHeader + reset + formatMessage,
        logging.WARNING: yellow + formatHeader + reset + formatMessage,
        logging.ERROR: red + formatHeader + reset + formatMessage,
        logging.CRITICAL: bold_red + formatHeader + reset + formatMessage
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
