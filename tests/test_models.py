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


@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        (
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                None
        ),
        (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                None
        ),
        (
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[0.33, 0.67, 1], [0.67, 0.83, 1],
                 [0.78, 0.89, 1]],
                None
        ),
        (
                [[np.nan, 2, 3], [0, 3, 1], [4, -2, 1]],
                [[0, 0.66, 1], [0, 1, 0.33], [1, 0, 0.25]],
                ValueError
        ),
        (
                [[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
                ValueError
        ),
        (
                str("A string"),
                None,
                TypeError
        ),
        (
                dict({"key": "value"}),
                None,
                TypeError
        ),
        (
                [[[1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2]]],
                None,
                ValueError
        ),
    ])
def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers.
       Test with a relative and absolute tolerance of 0.01."""
    from inflammation.models import patient_normalise

    if isinstance(test, list):
        test = np.array(test)

    if expect_raises is not None:
        with pytest.raises(expect_raises):
            _ = patient_normalise(test)

    else:
        result = patient_normalise(test)
        npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)