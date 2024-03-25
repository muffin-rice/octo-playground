from datetime import datetime
from enum import Enum


class LoggingLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


LEVELS = {
    LoggingLevel.DEBUG: 4,
    LoggingLevel.INFO: 3,
    LoggingLevel.WARN: 2,
    LoggingLevel.ERROR: 1,
}


class Logger:
    """Homebrew logger"""

    def __init__(
        self, namespace: str, logging_level: LoggingLevel = LoggingLevel.DEBUG, preamble_length = 50
    ):
        self.namespace = namespace
        self.logging_level = LEVELS[logging_level]
        self.preamble_length = preamble_length

    def print_format(self, level: str, message: str):
        preamble = f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]}][{level}] {self.namespace}: "
        if '\n' not in message:
            print("{:<45} {}".format(preamble, message))
        else:
            messages_split = message.split('\n')
            print("{:<45} {}".format(preamble, messages_split[0]))
            for message in messages_split[1:]:
                print("{:<45} {}".format("  ", message))

    def debug(self, message: str):
        if LEVELS[LoggingLevel.DEBUG] <= self.logging_level:
            self.print_format("DEBUG", message)

    def info(self, message: str):
        if LEVELS[LoggingLevel.INFO] <= self.logging_level:
            self.print_format("INFO", message)

    def warn(self, message: str):
        if LEVELS[LoggingLevel.WARN] <= self.logging_level:
            self.print_format("WARN", message)

    def error(self, message: str):
        if LEVELS[LoggingLevel.ERROR] <= self.logging_level:
            self.print_format("ERROR", message)
