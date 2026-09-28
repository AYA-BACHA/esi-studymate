"""Safe mathematical calculator tool for ESI StudyMate."""

import ast
import operator
import math
from typing import Union, Dict, Any

# Safe operators mapping
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe mathematical functions
SAFE_FUNCTIONS = {
    "ceil": math.ceil,
    "floor": math.floor,
    "sqrt": math.sqrt,
    "log2": math.log2,
    "log10": math.log10,
    "pow": math.pow,
    "round": round,
    "abs": abs,
    "min": min,
    "max": max,
}

# Safe constants
SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "KB": 1024,
    "MB": 1024 * 1024,
    "GB": 1024 * 1024 * 1024,
}


class SafeCalculator:
    """Safely evaluates mathematical expressions using AST parsing."""

    def _eval_node(self, node: ast.AST) -> Union[int, float]:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        elif isinstance(node, ast.Name):
            if node.id in SAFE_CONSTANTS:
                return SAFE_CONSTANTS[node.id]
            raise ValueError(f"Unknown constant or variable: '{node.id}'")

        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_type = type(node.op)
            if op_type in SAFE_OPERATORS:
                if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                    raise ZeroDivisionError("Division by zero")
                return SAFE_OPERATORS[op_type](left, right)
            raise ValueError(f"Unsupported operator: {op_type}")

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_type = type(node.op)
            if op_type in SAFE_OPERATORS:
                return SAFE_OPERATORS[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type}")

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
                func = SAFE_FUNCTIONS[node.func.id]
                args = [self._eval_node(arg) for arg in node.args]
                return func(*args)
            elif isinstance(node.func, ast.Attribute) and node.func.value.id == "math" and node.func.attr in SAFE_FUNCTIONS:
                func = SAFE_FUNCTIONS[node.func.attr]
                args = [self._eval_node(arg) for arg in node.args]
                return func(*args)
            raise ValueError(f"Disallowed or unknown function call")

        raise ValueError(f"Unsupported expression construct: {type(node).__name__}")

    def evaluate(self, expression: str) -> str:
        """Safely evaluate mathematical string."""
        expr_clean = expression.strip().replace("^", "**")
        try:
            tree = ast.parse(expr_clean, mode="eval")
            result = self._eval_node(tree.body)
            # Format cleanly
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return f"{result}"
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error evaluating expression '{expression}': {str(e)}"


calculator = SafeCalculator()


def calculate_tool(expression: str) -> Dict[str, Any]:
    """Calculate the result of a mathematical expression.
    
    Args:
        expression: A mathematical expression string, e.g. 'ceil(2500 / 512)' or '(2**32) / 4096'.
    """
    result = calculator.evaluate(expression)
    return {
        "expression": expression,
        "result": result,
    }
