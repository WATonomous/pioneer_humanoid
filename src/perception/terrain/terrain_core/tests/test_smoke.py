"""Smoke tests: basic checks that the package works at all."""

import terrain_core
from terrain_core import config


def test_package_imports():
    assert terrain_core is not None


def test_grid_shape_matches_spec():
    assert config.grid_shape() == (32, 20)


def test_total_cell_count():
    rows, cols = config.grid_shape()
    assert rows * cols == 640