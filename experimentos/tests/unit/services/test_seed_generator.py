"""Tests for seed_generator service."""

from experiments.services.seed_generator import generate_seed


class DescribeGenerateSeed:
    def it_returns_an_integer(self) -> None:
        seed = generate_seed()

        assert isinstance(seed, int)

    def it_returns_value_within_valid_range(self) -> None:
        seed = generate_seed()

        assert 0 <= seed < 2**32

    def it_generates_different_values_on_repeated_calls(self) -> None:
        # Generate multiple seeds to check randomness
        seeds = {generate_seed() for _ in range(100)}

        # With 100 random 32-bit integers, collisions are extremely unlikely
        assert len(seeds) > 90  # Allow some tolerance
