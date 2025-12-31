"""Tests for experiments.services.storage module."""

from pathlib import Path

from experiments.services.storage import (
    StorageService,
)


class DescribeStorageService:
    """Tests for the StorageService abstract base class."""

    class DescribeParseUri:
        """Tests for the parse_uri static method."""

        def it_parses_file_uri(self) -> None:
            """Verify file:// URIs are parsed correctly."""
            scheme, path = StorageService.parse_uri("file:///tmp/test.txt")
            assert scheme == "file"
            assert path == "/tmp/test.txt"

        def it_parses_s3_uri(self) -> None:
            """Verify s3:// URIs are parsed correctly."""
            scheme, path = StorageService.parse_uri("s3://bucket/path/to/file.txt")
            assert scheme == "s3"
            assert path == "bucket/path/to/file.txt"

        def it_parses_gs_uri(self) -> None:
            """Verify gs:// URIs are parsed correctly."""
            scheme, path = StorageService.parse_uri("gs://bucket/path/to/file.txt")
            assert scheme == "gs"
            assert path == "bucket/path/to/file.txt"

        def it_treats_plain_path_as_file(self) -> None:
            """Verify plain paths are treated as file:// URIs."""
            scheme, path = StorageService.parse_uri("/tmp/test.txt")
            assert scheme == "file"
            assert path == "/tmp/test.txt"

        def it_treats_relative_path_as_file(self) -> None:
            """Verify relative paths are treated as file:// URIs."""
            scheme, path = StorageService.parse_uri("data/test.txt")
            assert scheme == "file"
            assert path == "data/test.txt"

    class DescribeToUri:
        """Tests for the to_uri static method."""

        def it_converts_path_to_file_uri(self) -> None:
            """Verify Path objects are converted to file:// URIs."""
            uri = StorageService.to_uri(Path("/tmp/test.txt"))
            assert uri == "file:///tmp/test.txt"

        def it_converts_string_to_file_uri(self) -> None:
            """Verify string paths are converted to file:// URIs."""
            uri = StorageService.to_uri("/tmp/test.txt")
            assert uri == "file:///tmp/test.txt"

        def it_supports_s3_scheme(self) -> None:
            """Verify S3 scheme is supported."""
            uri = StorageService.to_uri("bucket/path/file.txt", scheme="s3")
            assert uri == "s3://bucket/path/file.txt"

        def it_supports_gs_scheme(self) -> None:
            """Verify GS scheme is supported."""
            uri = StorageService.to_uri("bucket/path/file.txt", scheme="gs")
            assert uri == "gs://bucket/path/file.txt"
