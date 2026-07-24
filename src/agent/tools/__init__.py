"""Agent tools: document search, calculator, SQL metadata, web search."""

from __future__ import annotations

from src.agent.tools.calculator import calculator
from src.agent.tools.search_documents import search_documents
from src.agent.tools.sql_metadata import sql_metadata_query
from src.agent.tools.web_search import web_search, web_search_available


def get_tools() -> list:
    """Return the active tool set (web search included only if configured)."""
    tools = [search_documents, calculator, sql_metadata_query]
    if web_search_available():
        tools.append(web_search)
    return tools


__all__ = [
    "calculator",
    "search_documents",
    "sql_metadata_query",
    "web_search",
    "web_search_available",
    "get_tools",
]
