"""
Result Pattern - Funktionale Fehlerbehandlung für Domain Operations.

Das Result Pattern ermöglicht eine explizite Behandlung von Erfolg und Fehler
ohne Exceptions zu werfen. Dies führt zu sauberem, vorhersagbarem Code.
"""

from typing import TypeVar, Generic, Optional, Callable, List, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


T = TypeVar('T')
E = TypeVar('E')
U = TypeVar('U')


@dataclass(frozen=True)
class Error:
    """Immutable error representation."""
    code: str
    message: str
    details: Optional[dict] = None

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class DomainError(Error):
    """Domain-specific error."""
    pass


class ValidationError(Error):
    """Validation error."""
    pass


class NotFoundError(Error):
    """Resource not found error."""
    pass


class BusinessRuleError(Error):
    """Business rule violation error."""
    pass


@dataclass
class Result(Generic[T]):
    """
    Result Monad für funktionale Fehlerbehandlung.

    Ermöglicht die Kapselung von Erfolg oder Fehler in einem einzigen Typ,
    ohne Exceptions zu werfen.

    Usage:
        result = Result.success(value)
        result = Result.failure(Error("CODE", "message"))

        if result.is_success:
            value = result.value
        else:
            error = result.error
    """
    _value: Optional[T] = None
    _error: Optional[Error] = None
    _errors: List[Error] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """Check if result is successful."""
        return self._error is None and not self._errors

    @property
    def is_failure(self) -> bool:
        """Check if result is a failure."""
        return not self.is_success

    @property
    def value(self) -> T:
        """Get the success value. Raises if result is failure."""
        if self.is_failure:
            raise ValueError(f"Cannot get value from failed result: {self.error}")
        return self._value

    @property
    def value_or_none(self) -> Optional[T]:
        """Get the success value or None if failure."""
        return self._value if self.is_success else None

    @property
    def error(self) -> Optional[Error]:
        """Get the error if result is failure."""
        return self._error or (self._errors[0] if self._errors else None)

    @property
    def errors(self) -> List[Error]:
        """Get all errors."""
        if self._error:
            return [self._error] + self._errors
        return self._errors

    @classmethod
    def success(cls, value: T) -> 'Result[T]':
        """Create a successful result."""
        return cls(_value=value)

    @classmethod
    def failure(cls, error: Union[Error, str]) -> 'Result[T]':
        """Create a failed result."""
        if isinstance(error, str):
            error = Error(code="ERROR", message=error)
        return cls(_error=error)

    @classmethod
    def failures(cls, errors: List[Error]) -> 'Result[T]':
        """Create a failed result with multiple errors."""
        return cls(_errors=errors)

    @classmethod
    def combine(cls, *results: 'Result') -> 'Result[List]':
        """
        Combine multiple results into one.

        Returns success with all values if all succeed,
        or failure with all errors if any fail.
        """
        errors = []
        values = []

        for result in results:
            if result.is_failure:
                errors.extend(result.errors)
            else:
                values.append(result.value)

        if errors:
            return cls.failures(errors)
        return cls.success(values)

    def map(self, func: Callable[[T], U]) -> 'Result[U]':
        """
        Map a function over the success value.

        If result is success, applies func to value.
        If result is failure, returns the failure unchanged.
        """
        if self.is_success:
            try:
                return Result.success(func(self._value))
            except Exception as e:
                return Result.failure(Error("MAP_ERROR", str(e)))
        return Result(_error=self._error, _errors=self._errors)

    def flat_map(self, func: Callable[[T], 'Result[U]']) -> 'Result[U]':
        """
        FlatMap a function over the success value.

        Like map, but the function itself returns a Result.
        """
        if self.is_success:
            try:
                return func(self._value)
            except Exception as e:
                return Result.failure(Error("FLATMAP_ERROR", str(e)))
        return Result(_error=self._error, _errors=self._errors)

    def on_success(self, func: Callable[[T], None]) -> 'Result[T]':
        """Execute function if result is success. Returns self for chaining."""
        if self.is_success:
            func(self._value)
        return self

    def on_failure(self, func: Callable[[Error], None]) -> 'Result[T]':
        """Execute function if result is failure. Returns self for chaining."""
        if self.is_failure:
            func(self.error)
        return self

    def get_or_default(self, default: T) -> T:
        """Get value or return default if failure."""
        return self._value if self.is_success else default

    def get_or_else(self, func: Callable[[], T]) -> T:
        """Get value or call function to get default if failure."""
        return self._value if self.is_success else func()

    def ensure(self, predicate: Callable[[T], bool], error: Error) -> 'Result[T]':
        """
        Ensure the value satisfies a predicate.

        Returns failure with error if predicate fails.
        """
        if self.is_failure:
            return self
        if not predicate(self._value):
            return Result.failure(error)
        return self

    def __repr__(self) -> str:
        if self.is_success:
            return f"Result.success({self._value!r})"
        return f"Result.failure({self.error!r})"


@dataclass
class ValidationResult:
    """
    Spezialfall von Result für Validierungen mit mehreren Fehlern.

    Akkumuliert alle Validierungsfehler statt beim ersten abzubrechen.
    """
    _errors: List[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return len(self._errors) == 0

    @property
    def is_invalid(self) -> bool:
        """Check if validation failed."""
        return len(self._errors) > 0

    @property
    def errors(self) -> List[ValidationError]:
        """Get all validation errors."""
        return self._errors.copy()

    def add_error(self, code: str, message: str, details: Optional[dict] = None) -> 'ValidationResult':
        """Add a validation error."""
        self._errors.append(ValidationError(code=code, message=message, details=details))
        return self

    def add_error_if(self, condition: bool, code: str, message: str) -> 'ValidationResult':
        """Add error only if condition is True."""
        if condition:
            self.add_error(code, message)
        return self

    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge another validation result into this one."""
        self._errors.extend(other.errors)
        return self

    def to_result(self, value: T) -> Result[T]:
        """Convert to Result type."""
        if self.is_valid:
            return Result.success(value)
        return Result.failures(list(self._errors))

    @classmethod
    def valid(cls) -> 'ValidationResult':
        """Create a valid result."""
        return cls()

    @classmethod
    def invalid(cls, code: str, message: str) -> 'ValidationResult':
        """Create an invalid result with one error."""
        result = cls()
        result.add_error(code, message)
        return result

    @classmethod
    def combine(cls, *results: 'ValidationResult') -> 'ValidationResult':
        """Combine multiple validation results."""
        combined = cls()
        for result in results:
            combined.merge(result)
        return combined
