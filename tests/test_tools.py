"""Tests for agent tools."""

import pytest
from tools import calculator, search_course_material


def test_calculator_basic():
    """Test calculator tool with basic arithmetic."""
    assert calculator.invoke("2 + 2") == "4"
    assert calculator.invoke("100 - 37") == "63"
    assert calculator.invoke("12 * 8") == "96"
    assert calculator.invoke("10 / 4") == "2.5"


def test_calculator_cs_math():
    """Test calculator tool with CS formulas like ceil and powers."""
    assert calculator.invoke("ceil(2500 / 512)") == "5"
    assert calculator.invoke("2**10") == "1024"
    assert calculator.invoke("2**32 / 4096") == "1048576.0"


def test_calculator_error_handling():
    """Test calculator handles division by zero safely."""
    res = calculator.invoke("10 / 0")
    assert "Error" in res


def test_search_tool_callable():
    """Test search_course_material tool runs and returns string."""
    res = search_course_material.invoke("interrupts")
    assert isinstance(res, str)
