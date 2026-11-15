"""Tests for the policy map size read from config/terrain.yaml."""
from pathlib import Path

import pytest

from terrain_core.config import PolicyMapConfig

TEAM_YAML = Path(__file__).resolve().parents[1] / "config" / "terrain.yaml"
WHOLE_NUMBER_TOL = 1e-9


def is_whole_number(value):
    return abs(value - round(value)) < WHOLE_NUMBER_TOL


@pytest.fixture
def policy():
    return PolicyMapConfig.from_yaml(TEAM_YAML)


# Settings ---------------------------------------------------------------------


def test_policy_map_is_16_by_10(policy):
    assert policy.shape() == (16, 10)


def test_sizes_and_resolution_are_positive(policy):
    assert policy.length_m > 0
    assert policy.width_m > 0
    assert policy.resolution_m > 0


@pytest.mark.parametrize("field", ["length_m", "width_m"])
def test_map_size_divides_evenly_into_cells(policy, field):
    size_m = getattr(policy, field)
    assert is_whole_number(size_m / policy.resolution_m), (
        f"{field} = {size_m} m is not a whole number of {policy.resolution_m} m cells"
    )


def test_first_index_is_forward_axis(policy):
    n_forward, n_sideways = policy.shape()
    assert n_forward > n_sideways


# Reading the file -------------------------------------------------------------


def test_from_yaml_reads_the_policy_section(tmp_path):
    path = tmp_path / "terrain.yaml"
    path.write_text("policy:\n  length_m: 2.0\n  width_m: 1.0\n  resolution_m: 0.5\n")
    assert PolicyMapConfig.from_yaml(path).shape() == (4, 2)
