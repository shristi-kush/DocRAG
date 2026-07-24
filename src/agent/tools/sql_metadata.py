"""SQL metadata tool: read-only queries over ingested-document metadata."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from src import metadata


@tool
def sql_metadata_query(query: str) -> str:
    """Run a read-only SQL SELECT over ingested-document metadata.

    Use this to answer questions about the documents that have been ingested,
    for example counts, page totals, or ingest dates.

    {schema}

    Only a single SELECT statement is permitted. Example:
    "SELECT COUNT(*) AS n FROM documents".
    """
    try:
        rows = metadata.run_select(query)
    except ValueError as exc:
        return f"Rejected query: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Query failed: {exc}"
    if not rows:
        return "No rows."
    return json.dumps(rows)


# Inject the live schema description into the tool docstring so the LLM sees it.
sql_metadata_query.description = sql_metadata_query.description.format(
    schema=metadata.schema_description()
)
