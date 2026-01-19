"""Service to generate random seeds for experiments."""

import random


def generate_seed() -> int:
    """Generates a random seed for reproducibility."""
    return random.randint(0, 2**32 - 1)
