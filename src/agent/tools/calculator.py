"""Calculator tool: safe arithmetic evaluation via numexpr."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result.

    Supports arithmetic (+, -, *, /, **, %) and parentheses, e.g.
    "0.15 * 3" or "(120 + 80) / 4". Use this for any calculation instead of
    doing arithmetic yourself.
    """
    import numexpr

    try:
        result = numexpr.evaluate(expression.strip(), local_dict={}, global_dict={})
        value = result.item()
    except Exception as exc:  # noqa: BLE001
        return f"Error: could not evaluate '{expression}' ({exc})."
    return str(value)
