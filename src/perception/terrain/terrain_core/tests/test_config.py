import pytest

from terrain_core import config

WHOLE_NUMBER_TOL = 1e-9


def cells_along(size_m):
    return size_m / config.RESOLUTION_M


def is_whole_number(value):
    return abs(value - round(value)) < WHOLE_NUMBER_TOL


# The spec -------------------------------------------------------------------


def test_grid_shape_is_32_by_20():
    assert config.grid_shape() == (32, 20)


def test_total_cell_count():
    n_x, n_y = config.grid_shape()
    assert n_x * n_y == 640


# Sanity of the numbers -------------------------------------------------------


def test_resolution_is_positive():
    assert config.RESOLUTION_M > 0


@pytest.mark.parametrize("size_m", [config.MAP_LENGTH_M, config.MAP_WIDTH_M])
def test_map_sizes_are_positive(size_m):
    assert size_m > 0


@pytest.mark.parametrize("size_m", [config.MAP_LENGTH_M, config.MAP_WIDTH_M])
def test_map_size_divides_evenly_into_cells(size_m):
    assert is_whole_number(cells_along(size_m)), (
        f"{size_m} m is not a whole number of {config.RESOLUTION_M} m cells"
    )


# Axis convention -------------------------------------------------------------


def test_first_index_is_forward_axis():
    n_x, n_y = config.grid_shape()
    assert n_x == round(cells_along(config.MAP_LENGTH_M))
    assert n_y == round(cells_along(config.MAP_WIDTH_M))
    assert n_x > n_y


# Check the checker -----------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [(32.0, True), (32.99999999999999, True), (32.4, False), (33.000001, False)],
)
def test_is_whole_number_helper(value, expected):
    assert is_whole_number(value) == expected