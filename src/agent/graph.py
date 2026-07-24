"""LangGraph tool-calling agent.

The agent node is the router: the LLM decides (via tool-calling) whether to
search the documents, run a calculation, query metadata, use web search, or
answer directly. A prebuilt ToolNode executes any requested tools and loops
back until the model produces a final answer.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.tools import get_tools
from src.llm import get_llm

logger = logging.getLogger(__name__)

_SYSTEM = SystemMessage(
    content=(
        "You are DocRAG, an assistant that answers questions about the user's "
        "uploaded documents. Use the available tools:\n"
        "- search_documents: search the uploaded documents. PREFER THIS FIRST "
        "for any question about document content.\n"
        "- calculator: evaluate arithmetic. Always use it for math rather than "
        "computing in your head.\n"
        "- sql_metadata_query: read-only SQL about which documents have been "
        "ingested (counts, page totals, ingest dates).\n"
        "- web_search (only if available): public-web fallback, used ONLY when "
        "the documents do not contain the answer.\n\n"
        "Ground answers in tool results. If the documents lack the answer and "
        "web search is unavailable, say you do not know based on the uploaded "
        "documents. Keep answers concise."
    )
)

_agent = None


def _build():
    llm = get_llm()
    tools = get_tools()
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState) -> dict:
        response = llm_with_tools.invoke([_SYSTEM] + state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", END])
    builder.add_edge("tools", "agent")
    return builder.compile()


def get_agent():
    global _agent
    if _agent is None:
        _agent = _build()
        logger.info("Built LangGraph agent with %d tools.", len(get_tools()))
    return _agent


def answer(message: str) -> str:
    """Run the agent to completion and return the final text answer."""
    if not message or not message.strip():
        raise ValueError("Message must not be empty")
    result = get_agent().invoke(
        {"messages": [HumanMessage(content=message.strip())]}
    )
    final = result["messages"][-1]
    return getattr(final, "content", str(final))
