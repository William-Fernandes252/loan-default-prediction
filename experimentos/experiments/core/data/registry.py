"""Transformer registry for automatic registration of data transformers.

This module provides a registration system that allows data transformers
to self-register, eliminating the need to modify factory code when
adding new datasets. This adheres to the Open/Closed Principle.

Example:
    ```python
    @register_transformer("my_dataset")
    class MyDatasetTransformer(BaseDataTransformer):
        ...
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from experiments.core.data.base import BaseDataTransformer

T = TypeVar("T", bound="BaseDataTransformer")

# Global registry mapping dataset IDs to transformer classes
_TRANSFORMER_REGISTRY: dict[str, type[BaseDataTransformer]] = {}


def register_transformer(dataset_id: str) -> type[T]:
    """Decorator to register a transformer for a dataset.

    This decorator automatically adds the transformer class to the
    global registry, making it available to the factory without
    requiring manual registration.

    Args:
        dataset_id: The unique identifier for the dataset (e.g., "taiwan_credit").

    Returns:
        The decorator function.

    Example:
        ```python
        @register_transformer("taiwan_credit")
        class TaiwanCreditTransformer(BaseDataTransformer):
            ...
        ```

    Raises:
        ValueError: If a transformer is already registered for the dataset_id.
    """

    def decorator(cls: type[T]) -> type[T]:
        if dataset_id in _TRANSFORMER_REGISTRY:
            raise ValueError(
                f"Transformer already registered for dataset '{dataset_id}': "
                f"{_TRANSFORMER_REGISTRY[dataset_id].__name__}"
            )
        _TRANSFORMER_REGISTRY[dataset_id] = cls
        return cls

    return decorator


def get_transformer_registry() -> dict[str, type[BaseDataTransformer]]:
    """Get a copy of the current transformer registry.

    Returns:
        A dictionary mapping dataset IDs to transformer classes.
    """
    return _TRANSFORMER_REGISTRY.copy()


def get_transformer(dataset_id: str) -> type[BaseDataTransformer] | None:
    """Get a transformer class for a dataset ID.

    Args:
        dataset_id: The dataset identifier.

    Returns:
        The transformer class, or None if not registered.
    """
    return _TRANSFORMER_REGISTRY.get(dataset_id)


def clear_registry() -> None:
    """Clear the transformer registry.

    This is primarily useful for testing.
    """
    _TRANSFORMER_REGISTRY.clear()
