"""Project-wide exception hierarchy."""


class MT5BotError(Exception):
    """Base exception for all trading bot errors."""


class ModelError(MT5BotError):
    """Raised when a model is used incorrectly (not trained, bad features, etc.)."""


class DataError(MT5BotError):
    """Raised when input data is invalid, missing, or has lookahead contamination."""


class ConfigError(MT5BotError):
    """Raised when config.yaml is missing a required key or has an invalid value."""


class BrokerError(MT5BotError):
    """Raised when the MT5 bridge returns an error or is unreachable."""
