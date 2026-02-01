"""
Shared error types for core services.
"""


class ValidationIssue(ValueError):
    def __init__(
        self,
        message: str,
        field: str = "unknown",
        error_type: str = "invalid",
        error_code: str | None = None,
        data: dict | None = None,
    ):
        super().__init__(message)
        self.field = field
        self.error_type = error_type
        self.error_code = error_code
        self.data = data


class EmbeddingProviderError(RuntimeError):
    """Raised when the embedding provider is unavailable."""
