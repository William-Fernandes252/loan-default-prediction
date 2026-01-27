"""Tests for seed_generator service."""

from experiments.services.seed_generator import generate_seed


class DescribeGenerateSeed:
    def it_returns_an_integer(self) -> None:
        result = generate_seed()

        assert isinstance(result, int)

    def it_returns_value_within_32bit_range(self) -> None:
        result = generate_seed()

        assert 0 <= result < 2**32

    def it_generates_unique_values_across_calls(self) -> None:
        seeds = {generate_seed() for _ in range(100)}

        # With 100 random 32-bit integers, collisions are extremely unlikely
        assert len(seeds) > 90
