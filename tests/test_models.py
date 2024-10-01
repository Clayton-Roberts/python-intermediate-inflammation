"""Tests for statistics functions within the Model layer."""
# pylint: disable=C0415
import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[1, 2], [3, 4], [5, 6]], [5, 6]),
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[0, 1, 2], [3, 4, 10], [6, 7, 8]], [6, 7, 10]),
    ])
def test_daily_max(test, expected):
    """Test that max function works various arrays."""
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[1, 2], [3, 4], [5, 6]], [1, 2]),
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[0, 1, 2], [3, 4, -5], [6, 7, 8]], [0, 1, -5]),
        ([[0, -8, 2], [3, 4, -5], [6, 7, 8]], [0, -8, -5]),
    ])
def test_daily_min(test, expected):
    """Test that min function works various arrays."""
    from inflammation.models import daily_min
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))


def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        daily_min([['Hello', 'there'], ['General', 'Kenobi']])
