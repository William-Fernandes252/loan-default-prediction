"""Tests for the transformer registry module."""

import pytest

from experiments.core.data.base import BaseDataTransformer
from experiments.core.data.registry import (
    clear_registry,
    get_transformer,
    get_transformer_registry,
    register_transformer,
)


class DescribeRegisterTransformer:
    """Tests for register_transformer decorator."""

    def it_registers_transformer_class(self) -> None:
        """Verify decorator registers a transformer."""
        clear_registry()

        @register_transformer("test_dataset")
        class TestTransformer(BaseDataTransformer):
            @property
            def dataset_name(self) -> str:
                return "test_dataset"

            def _apply_transformations(self, df):
                return df

        registry = get_transformer_registry()
        assert "test_dataset" in registry
        assert registry["test_dataset"] == TestTransformer

    def it_returns_decorated_class(self) -> None:
        """Verify decorator returns the original class."""
        clear_registry()

        @register_transformer("test_dataset_2")
        class TestTransformer(BaseDataTransformer):
            @property
            def dataset_name(self) -> str:
                return "test_dataset_2"

            def _apply_transformations(self, df):
                return df

        # Should be able to instantiate the class normally
        instance = TestTransformer(use_gpu=False)
        assert isinstance(instance, BaseDataTransformer)

    def it_raises_on_duplicate_registration(self) -> None:
        """Verify raises ValueError when registering same dataset twice."""
        clear_registry()

        @register_transformer("duplicate_dataset")
        class FirstTransformer(BaseDataTransformer):
            @property
            def dataset_name(self) -> str:
                return "duplicate_dataset"

            def _apply_transformations(self, df):
                return df

        with pytest.raises(ValueError, match="already registered"):

            @register_transformer("duplicate_dataset")
            class SecondTransformer(BaseDataTransformer):
                @property
                def dataset_name(self) -> str:
                    return "duplicate_dataset"

                def _apply_transformations(self, df):
                    return df


class DescribeGetTransformerRegistry:
    """Tests for get_transformer_registry function."""

    def it_returns_copy_of_registry(self) -> None:
        """Verify returns a copy, not the original registry."""
        clear_registry()

        @register_transformer("test_dataset_3")
        class TestTransformer(BaseDataTransformer):
            @property
            def dataset_name(self) -> str:
                return "test_dataset_3"

            def _apply_transformations(self, df):
                return df

        registry1 = get_transformer_registry()
        registry2 = get_transformer_registry()

        # Should be equal but not the same object
        assert registry1 == registry2
        assert registry1 is not registry2

    def it_modifications_dont_affect_original(self) -> None:
        """Verify modifying returned dict doesn't affect registry."""
        clear_registry()

        @register_transformer("test_dataset_4")
        class TestTransformer(BaseDataTransformer):
            @property
            def dataset_name(self) -> str:
                return "test_dataset_4"

            def _apply_transformations(self, df):
                return df

        registry_copy = get_transformer_registry()
        registry_copy["fake_dataset"] = TestTransformer

        # Original registry should not have fake_dataset
        original = get_transformer_registry()
        assert "fake_dataset" not in original
        assert "test_dataset_4" in original


class DescribeGetTransformer:
    """Tests for get_transformer function."""

    def it_returns_registered_transformer(self) -> None:
        """Verify returns the correct transformer class."""
        clear_registry()

        @register_transformer("test_dataset_5")
        class TestTransformer(BaseDataTransformer):
            @property
            def dataset_name(self) -> str:
                return "test_dataset_5"

            def _apply_transformations(self, df):
                return df

        transformer = get_transformer("test_dataset_5")
        assert transformer == TestTransformer

    def it_returns_none_for_unregistered_dataset(self) -> None:
        """Verify returns None for unknown dataset IDs."""
        clear_registry()

        transformer = get_transformer("nonexistent_dataset")
        assert transformer is None


class DescribeClearRegistry:
    """Tests for clear_registry function."""

    def it_removes_all_registrations(self) -> None:
        """Verify clears all transformer registrations."""
        clear_registry()

        @register_transformer("test_dataset_6")
        class TestTransformer(BaseDataTransformer):
            @property
            def dataset_name(self) -> str:
                return "test_dataset_6"

            def _apply_transformations(self, df):
                return df

        assert len(get_transformer_registry()) > 0

        clear_registry()

        assert len(get_transformer_registry()) == 0


class DescribeTransformerRegistrationIntegration:
    """Integration tests for the registration system."""

    def it_allows_adding_new_transformers_without_code_changes(self) -> None:
        """Verify new transformers can be added by just creating them."""
        clear_registry()

        # Simulate adding a new dataset transformer
        @register_transformer("new_fancy_dataset")
        class NewFancyTransformer(BaseDataTransformer):
            @property
            def dataset_name(self) -> str:
                return "new_fancy_dataset"

            def _apply_transformations(self, df):
                return df

        # Should be immediately available in registry
        registry = get_transformer_registry()
        assert "new_fancy_dataset" in registry

        # Should be retrievable
        transformer_cls = get_transformer("new_fancy_dataset")
        assert transformer_cls == NewFancyTransformer

        # Should be instantiable
        instance = transformer_cls(use_gpu=False)
        assert isinstance(instance, BaseDataTransformer)
