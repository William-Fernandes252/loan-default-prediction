from typing import NamedTuple


class Choice(NamedTuple):
    """Entity used in the experiments."""

    id: str
    display_name: str

    def __repr__(self):
        return self.id

    def __str__(self):
        return self.id
