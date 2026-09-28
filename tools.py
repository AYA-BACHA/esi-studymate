"""Tools for the ESI StudyMate Agent.

Uses the simple @tool decorator from LangChain.
"""

import math
from langchain_core.tools import tool
from rag import search_documents


@tool
def search_course_material(query: str) -> str:
    """Search the uploaded university course materials for relevant concepts, slides, and definitions."""
    return search_documents(query)


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations. Useful for computer science problems like memory blocks or page sizes.
    
    Example expressions: '2500 / 512', 'ceil(2500 / 512)', '2**32 / 4096'.
    """
    safe_dict = {
        "ceil": math.ceil,
        "floor": math.floor,
        "sqrt": math.sqrt,
        "log2": math.log2,
        "round": round,
        "abs": abs,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        # Clean expression
        clean_expr = expression.strip().replace("^", "**")
        result = eval(clean_expr, {"__builtins__": {}}, safe_dict)
        return str(result)
    except Exception as e:
        return f"Error calculating: {e}"
