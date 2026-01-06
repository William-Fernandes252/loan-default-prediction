"""Transformer registry for automatic registration of data transformers.

This module provides a registration system that allows data transformers
to self-register, eliminating the need to modify factory code when
adding new datasets.

Example:
    ```python
    @register_transformer("my_dataset")
    class MyDatasetTransformer(BaseDataTransformer):
        ...
    ```
"""

from __future__ import annotations

from typing import Callable, Mapping

from experiments.core.data_new.transformer import Transformer

# Global registry mapping dataset IDs to transformers
_TRANSFORMER_REGISTRY: dict[str, Transformer] = {}

type TransformerRegistry = Mapping[str, Transformer]
"""Mapping of dataset IDs to transformer instances."""


def register_transformer(dataset_id: str) -> Callable[[Transformer], Transformer]:
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

    def decorator(t: Transformer) -> Transformer:
        if dataset_id in _TRANSFORMER_REGISTRY:
            raise ValueError(f"Transformer already registered for dataset '{dataset_id}'.")
        _TRANSFORMER_REGISTRY[dataset_id] = t
        return t

    return decorator


def get_transformer_registry() -> TransformerRegistry:
    """Get a copy of the current transformer registry.

    Returns:
        A dictionary mapping dataset IDs to transformer classes.
    """
    return _TRANSFORMER_REGISTRY.copy()


def get_transformer(dataset_id: str) -> Transformer | None:
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
