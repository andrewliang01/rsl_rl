from __future__ import annotations

import math

import pytest

from rsl_rl.utils.logger import format_duration


@pytest.mark.parametrize(
    ("seconds", "expected"),
    (
        (0.0, "00:00:00"),
        (59.9, "00:00:59"),
        (3_661.2, "01:01:01"),
        (86_399.9, "23:59:59"),
        (86_400.0, "1d 00:00:00"),
        (109_704.0, "1d 06:28:24"),
        (2 * 86_400 + 7, "2d 00:00:07"),
    ),
)
def test_format_duration_does_not_wrap_after_24_hours(
    seconds: float, expected: str
) -> None:
    assert format_duration(seconds) == expected


@pytest.mark.parametrize("seconds", (-1.0, math.inf, -math.inf, math.nan))
def test_format_duration_rejects_invalid_values(seconds: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        format_duration(seconds)
