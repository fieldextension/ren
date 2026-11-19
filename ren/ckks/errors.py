"""Exceptions raised by Ren's CKKS implementation."""

from typing import Any


class SchemeException(Exception):
    """
    Base exception class for all scheme-level errors.

    All scheme-level exceptions in this library should inherit from this class.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args)
        for key, value in kwargs.items():
            setattr(self, key, value)


class ParameterError(SchemeException):
    """Base class for parameter-related errors."""

    pass


class ParameterMismatchError(ParameterError):
    """
    Raised when cryptographic parameters don't match between operations.

    Examples:
        - Polynomial degree mismatch
        - Modulus mismatch
        - Different parameter sets between ciphertexts
    """

    pass


class InvalidParameterError(ParameterError):
    """
    Raised when a parameter value is invalid for the operation.

    Examples:
        - Negative modulus
        - Non-power-of-two polynomial degree
        - Invalid base for decomposition
    """

    pass


class NoiseOverflowError(SchemeException):
    """
    Raised when noise in ciphertext is too large.

    Examples:
        - Decryption not guaranteed to be correct due to excessive noise
        - Bootstrapping not guaranteed to work due to excessive noise
    """

    pass
