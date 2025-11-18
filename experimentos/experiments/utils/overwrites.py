from __future__ import annotations

from typing import Callable, Iterable, Sequence, TypeVar

import typer

T = TypeVar("T")

ExistsFn = Callable[[T], bool]
PromptFn = Callable[[T], str]
SkipFn = Callable[[T], None]


def confirm_overwrite(prompt: str) -> bool:
    """Ask the user to confirm an overwrite operation."""

    return typer.confirm(prompt, default=False)


def filter_items_for_processing(
    items: Sequence[T] | Iterable[T],
    *,
    exists_fn: ExistsFn,
    prompt_fn: PromptFn,
    force: bool,
    on_skip: SkipFn | None = None,
) -> list[T]:
    """Return subset of items approved for processing, prompting when needed."""

    ready: list[T] = []
    for item in items:
        should_prompt = exists_fn(item) and not force
        if should_prompt:
            prompt = prompt_fn(item)
            if confirm_overwrite(prompt):
                ready.append(item)
            else:
                if on_skip is not None:
                    on_skip(item)
        else:
            ready.append(item)

    return ready
