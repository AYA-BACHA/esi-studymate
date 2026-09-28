"""Test 1: Safe Mathematical Calculator Tool."""

import pytest
from app.tools.calculator import calculator, calculate_tool


def test_basic_arithmetic():
    """Test standard arithmetic operations."""
    assert calculator.evaluate("2 + 2") == "4"
    assert calculator.evaluate("100 - 37") == "63"
    assert calculator.evaluate("12 * 8") == "96"
    assert calculator.evaluate("10 / 4") == "2.5"
    assert calculator.evaluate("17 // 5") == "3"
    assert calculator.evaluate("17 % 5") == "2"


def test_powers_and_parentheses():
    """Test exponentiation and operator precedence."""
    assert calculator.evaluate("2 ** 10") == "1024"
    assert calculator.evaluate("(2 + 3) * 4") == "20"
    assert calculator.evaluate("2**32 / 4096") == "1048576"


def test_math_functions():
    """Test safe mathematical functions like ceil, floor, log2."""
    assert calculator.evaluate("ceil(2500 / 512)") == "5"
    assert calculator.evaluate("floor(2500 / 512)") == "4"
    assert calculator.evaluate("log2(1024)") == "10"
    assert calculator.evaluate("sqrt(144)") == "12"


def test_division_by_zero():
    """Test safe error handling on division by zero."""
    result = calculator.evaluate("10 / 0")
    assert "Division by zero" in result


def test_disallowed_syntax_safety():
    """Test that unauthorized constructs (e.g. imports, system calls) are rejected."""
    result = calculator.evaluate("__import__('os').system('ls')")
    assert "Error" in result or "Disallowed" in result


def test_calculate_tool_dict_output():
    """Test calculate_tool wrapper output schema."""
    res = calculate_tool("ceil(2500 / 512)")
    assert isinstance(res, dict)
    assert res["expression"] == "ceil(2500 / 512)"
    assert res["result"] == "5"
