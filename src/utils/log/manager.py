# Standard library imports
import os
import logging
from typing import Annotated, Optional
from logging.handlers import RotatingFileHandler

# Third-party imports
import colorlog
import colorama

# Initialize colorama for Windows OS
colorama.init()


class MultiLineColorFormatter(colorlog.ColoredFormatter):
    """
    A specialized formatter that extends ``colorlog.ColoredFormatter`` to
    format log messages in a multi-line format with color-coded output.

    This class is designed to display logs with multiple lines, including
    timestamp, logger name, log level, and log message in a visually
    appealing way.

    Examples
    --------
    >>> import logging
    >>> from src.utils.log.manager import MultiLineColorFormatter
    >>> formatter = MultiLineColorFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    >>> console_handler = logging.StreamHandler()
    >>> console_handler.setFormatter(formatter)
    >>> logger_ = logging.getLogger("example_logger")
    >>> logger.addHandler(console_handler)
    >>> logger.info("Sample log message")
    """

    def __init__(
            self,
            datefmt: Annotated[Optional[str], "Date format string for timestamps"] = None
    ) -> None:
        """
        Initialize the MultiLineColorFormatter.

        Parameters
        ----------
        datefmt : str, optional
            The date format string for formatting timestamps. Defaults
            to None.

        Raises
        ------
        TypeError
            If ``datefmt`` is not a string or ``None``.
        """
        if datefmt is not None and not isinstance(datefmt, str):
            raise TypeError("Expected str or None for parameter 'datefmt'.")

        fmt = (
            "%(white)s%(asctime)s%(reset)s\n\n"
            "=== %(name)s ===\n\n"
            "%(log_color)s[%(levelname)s]%(reset)s\n\n"
            "%(message)s\n"
        )

        super().__init__(
            fmt=fmt,
            datefmt=datefmt,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red',
            },
            secondary_log_colors={
                'white': {
                    'DEBUG': 'white',
                    'INFO': 'white',
                    'WARNING': 'white',
                    'ERROR': 'white',
                    'CRITICAL': 'white',
                }
            },
        )


class MultiLineFileFormatter(logging.Formatter):
    """
    A specialized formatter for file logging that formats log messages in
    a multi-line format suitable for file output.

    This formatter displays the timestamp, logger name, log level, and
    message in multiple lines, improving readability in log files.

    Examples
    --------
    >>> import logging
    >>> from src.utils.log.manager import MultiLineColorFormatter
    >>> file_handler = logging.FileHandler('example.log')
    >>> file_handler.setFormatter(MultiLineFileFormatter(
    ...     datefmt="%Y-%m-%d %H:%M:%S")
    ... )
    >>> logger_test = logging.getLogger("example_logger")
    >>> logger.addHandler(file_handler)
    >>> logger.info("Sample log message")
    """

    def __init__(
            self,
            datefmt: Annotated[Optional[str], "Date format string for timestamps"] = None
    ) -> None:
        """
        Initialize the MultiLineFileFormatter.

        Parameters
        ----------
        datefmt : str, optional
            The date format string for formatting timestamps. Defaults
            to None.

        Raises
        ------
        TypeError
            If ``datefmt`` is not a string or ``None``.
        """
        if datefmt is not None and not isinstance(datefmt, str):
            raise TypeError("Expected str or None for parameter 'datefmt'.")

        super().__init__(datefmt=datefmt)
        self.datefmt = datefmt
        self._fmt = (
            "%(asctime)s\n\n"
            "=== %(name)s ===\n\n"
            "[%(levelname)s]\n\n"
            "%(message)s\n"
        )

    def format(
            self,
            record: Annotated[logging.LogRecord, "Logging record"]
    ) -> Annotated[str, "Formatted log message"]:
        """
        Format the specified log record into a multi-line string.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format.

        Returns
        -------
        str
            The formatted log message as a multi-line string.

        Examples
        --------
        >>> import logging
        >>> from src.utils.log.manager import MultiLineColorFormatter
        >>> formatter = MultiLineFileFormatter()
        >>> record_test = logging.LogRecord(
        ...     name="test",
        ...     level=logging.INFO,
        ...     pathname="",
        ...     lineno=0,
        ...     msg="Sample message",
        ...     args=None,
        ...     exc_info=None
        ... )
        >>> formatted_message = formatter.format(record)
        >>> print(formatted_message)
        2025-01-27 12:34:56

        === test ===

        [INFO]

        Sample message
        """
        record_asctime = self.formatTime(record, self.datefmt)
        return self._fmt % {
            'asctime': record_asctime,
            'name': record.name,
            'levelname': record.levelname,
            'message': record.getMessage(),
        }


class LoggerManager:
    """
    Manages the creation and configuration of loggers for console and
    file output using rotating file handlers.

    This class simplifies the setup of log handlers, formatters, and
    logging levels for both console and file logging.

    Parameters
    ----------
    log_dir : str, optional
        The directory where log files will be stored. Defaults to
        ``.log``.
    log_file : str, optional
        The name of the log file. Defaults to ``heartbeat.log``.
    logger_name : str, optional
        The name of the logger. Defaults to ``HeartbeatLog``.
    console_level : int, optional
        The logging level for console output. Defaults to
        ``logging.DEBUG``.
    file_level : int, optional
        The logging level for file output. Defaults to
        ``logging.INFO``.
    max_bytes : int, optional
        The maximum file size in bytes before rotation occurs. Defaults
        to 5,000,000.
    backup_count : int, optional
        The number of rotated log files to keep. Defaults to 5.
    verbose : bool, optional
        Whether to enable console logging. Defaults to True.

    Attributes
    ----------
    logger : logging.Logger
        The configured logger instance.

    Methods
    -------
    get_logger()
        Retrieves the configured logger instance.

    Examples
    --------
    >>> manager = LoggerManager(log_file="example.log",
    ...                         logger_name="ExampleLogger")
    >>> logger_test = manager.get_logger()
    >>> logger.info("Sample log message")
    """

    def __init__(
            self,
            log_dir: Annotated[str, "Directory for log files"] = ".logs",
            log_file: Annotated[str, "Name of the log file"] = "heartbeat.log",
            logger_name: Annotated[str, "Logger name"] = "HeartbeatLog",
            console_level: Annotated[int, "Console log level"] = logging.DEBUG,
            file_level: Annotated[int, "File log level"] = logging.INFO,
            max_bytes: Annotated[int, "Max file size for rotation"] = 5_000_000,
            backup_count: Annotated[int, "Number of backup files to keep"] = 5,
            verbose: Annotated[bool, "Enable console logging"] = True
    ) -> None:
        """
        Initialize the LoggerManager with the specified configuration.

        Parameters
        ----------
        log_dir : str, optional
            The directory where log files will be stored. Defaults to
            ``.logs``.
        log_file : str, optional
            The name of the log file. Defaults to ``heartbeat.log``.
        logger_name : str, optional
            The name of the logger. Defaults to ``HeartbeatLog``.
        console_level : int, optional
            The logging level for console output. Defaults to
            ``logging.DEBUG``.
        file_level : int, optional
            The logging level for file output. Defaults to
            ``logging.INFO``.
        max_bytes : int, optional
            The maximum file size in bytes before rotation. Defaults
            to 5,000,000.
        backup_count : int, optional
            The number of rotated log files to keep. Defaults to 5.
        verbose : bool, optional
            Whether to enable console logging. Defaults to True.

        Raises
        ------
        TypeError
            If any of the provided parameters have an invalid type.
        """
        if not isinstance(log_dir, str):
            raise TypeError("Expected str for parameter 'log_dir'.")
        if not isinstance(log_file, str):
            raise TypeError("Expected str for parameter 'log_file'.")
        if not isinstance(logger_name, str):
            raise TypeError("Expected str for parameter 'logger_name'.")
        if not isinstance(console_level, int):
            raise TypeError("Expected int for parameter 'console_level'.")
        if not isinstance(file_level, int):
            raise TypeError("Expected int for parameter 'file_level'.")
        if not isinstance(max_bytes, int):
            raise TypeError("Expected int for parameter 'max_bytes'.")
        if not isinstance(backup_count, int):
            raise TypeError("Expected int for parameter 'backup_count'.")
        if not isinstance(verbose, bool):
            raise TypeError("Expected bool for parameter 'verbose'.")

        self.log_dir = log_dir
        self.log_file = log_file
        self.logger_name = logger_name
        self.console_level = console_level
        self.file_level = file_level
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.verbose = verbose

        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self) -> None:
        """
        Create and configure the file and console handlers for the logger.

        Raises
        ------
        OSError
            If the log directory cannot be created.
        """
        os.makedirs(self.log_dir, exist_ok=True)
        log_file_path = os.path.join(self.log_dir, self.log_file)

        file_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        file_handler.setLevel(self.file_level)
        file_formatter = MultiLineFileFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.console_level)
            console_formatter = MultiLineColorFormatter(
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

    def get_logger(
            self
    ) -> Annotated[logging.Logger, "Configured logger instance"]:
        """
        Retrieve the configured logger instance.

        Returns
        -------
        logging.Logger
            The logger configured by this LoggerManager.

        Examples
        --------
        >>> manager = LoggerManager(logger_name="TestLogger")
        >>> logger_test = manager.get_logger()
        >>> logger.info("Hello, world!")
        """
        return self.logger


if __name__ == "__main__":
    logger_manager = LoggerManager(
        log_file="example.log",
        logger_name="ExampleLogger",
        backup_count=3,
    )

    logger = logger_manager.get_logger()

    logger.debug("This is a DEBUG-level log message.")
    logger.info("This is an INFO-level log message.")
    logger.warning("This is a WARNING-level log message.")
    logger.error("This is an ERROR-level log message.")
    logger.critical("This is a CRITICAL-level log message.")
