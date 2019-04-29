#!/usr/bin/env python3
# coding: utf8

"""
Tests JSON(C) configuration.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"


import os

from unmix.source.configuration import Configuration


def test_configuration_initialize():
    current_path = os.path.dirname(__file__)
    config_file = os.path.join(os.path.dirname(__file__), 'test.jsonc')
    Configuration.initialize(config_file, current_path, create_output=False)
    assert Configuration.get("test") == "test-root"
    assert Configuration.get("level1.level2.level3") == "test-level3"
    assert Configuration.get("level1").level2.level3 == "test-level3"
    assert type(Configuration.get("float_implicit")) == float
    assert type(Configuration.get("float_explicit")) == float
    assert type(Configuration.get("int_implicit")) == int
    assert type(Configuration.get("int_explicit")) == int
    assert type(Configuration.get("bool_implicit")) == bool
    assert type(Configuration.get("bool_explicit")) == bool
    test_path = Configuration.get_path("path")
    assert os.path.isabs(test_path)
    assert test_path.startswith(current_path)
    assert Configuration.get("nonexistent", optional=True) is None
    assert Configuration.get("nonexistent", optional=True, default=42) == 42
    try:
        Configuration.get("nonexistent", optional=False)
        assert False
    except:
        assert True


if __name__ == "__main__":
    test_configuration_initialize()
    print("Test run successful.")
