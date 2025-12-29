class StorageError(Exception):
    """Raised when there is an error with storage operations."""

    def __init__(self, uri: str, reason: str):
        self.uri = uri
        self.reason = reason
        super().__init__(f"Storage error for '{uri}': {reason}")


class FileDoesNotExistError(StorageError):
    """Raised when a requested file does not exist in storage."""

    def __init__(self, uri: str):
        super().__init__(uri, "File does not exist")
