"""Definitions related to pipeline context and execution."""

from typing import TypeVar

Context = TypeVar("Context")
"""Context is a generic type variable representing the context in a pipeline.

It is optional and can be any type, allowing for flexibility in defining different pipeline contexts.

The context is immutable, meaning that pipeline steps should not modify the context directly, and it should be used only for things like configuration or shared resources.
"""
