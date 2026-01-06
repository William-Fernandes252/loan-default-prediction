"""Definitions for pipeline state types."""

from typing import TypeVar

State = TypeVar("State")
"""State is a generic type variable representing the state in a pipeline.

It can be any type, allowing for flexibility in defining different pipeline states. 

The state is mutable, meaning that pipeline steps can modify and return updated versions of the state as needed.
"""
