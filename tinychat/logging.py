import logging
import sys
from typing import Literal, Optional

from loguru import logger
from loguru._handler import Handler


class InterceptHandler(logging.Handler):
    """
    Default handler from examples in loguru documentation.

    This handler intercepts all log requests and
    passes them to loguru.

    For more info see:
    https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    """

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        """
        Propagates logs to loguru.

        :param record: record to log.
        """
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while (
            frame.f_code.co_filename == logging.__file__
            or frame.f_code.co_filename == __file__
            or "sentry_sdk/integrations" in frame.f_code.co_filename
        ):
            frame = frame.f_back  # type: ignore
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


def configure_intercepter() -> None:
    """
    Configures the logging system to intercept log messages.

    This function sets up an InterceptHandler instance as the main handler for the root logger.
    It sets the logging level to INFO, meaning that all messages with severity INFO and above will be handled.

    It then iterates over all the loggers in the logging system. If a logger's name starts with "uvicorn.",
    it removes all handlers from that logger. This is done to prevent uvicorn's default logging configuration
    from interfering with our custom configuration.

    Finally, it sets the InterceptHandler instance as the sole handler for the "uvicorn" and "uvicorn.access" loggers.
    This ensures that all log messages from uvicorn and its access logger are intercepted by our custom handler.
    """
    intercept_handler = InterceptHandler()
    logging.basicConfig(handlers=[intercept_handler], level=logging.INFO)

    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith("uvicorn."):
            logging.getLogger(logger_name).handlers = []

    logging.getLogger("uvicorn").handlers = [intercept_handler]
    logging.getLogger("uvicorn.access").handlers = [intercept_handler]


def configure_pretty_logging(debug_level: int = logging.DEBUG) -> None:
    """
    Configures the logging system to output pretty logs.

    This function enables the 'tinychat' logger, sets up an intercept handler to
    capture logs from the standard logging module, removes all existing handlers
    from the 'loguru' logger, and adds a new handler that outputs to stdout with
    pretty formatting (colored, not serialized, no backtrace or diagnosis information).

    Args:
        debug_level: The logging level to use. Should be one of the following:
            10 (DEBUG), 20 (INFO), 30 (WARNING), 40 (ERROR), or 50 (CRITICAL).
    """
    logger.enable("tinychat")

    configure_intercepter()

    logger.remove()
    logger.add(
        sys.stdout,
        level=debug_level,
        backtrace=False,
        diagnose=False,
        serialize=False,
        colorize=True,
    )
