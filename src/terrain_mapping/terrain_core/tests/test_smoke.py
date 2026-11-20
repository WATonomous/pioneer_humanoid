"""Smoke tests: basic checks that the package works at all."""

import terrain_core


def test_package_imports():
    assert terrain_core is not None
