"""Compatibility shims for running RAGAS on LangChain 1.x.

RAGAS 0.4.x still hard-imports legacy Vertex AI classes at module load
(``langchain_community.chat_models.vertexai.ChatVertexAI`` and
``langchain_community.llms.VertexAI``) which were removed when
``langchain-community`` was reorganised for LangChain 1.x. Because DocRAG
evaluates entirely with a local Ollama model, those classes are never used, so
we register harmless placeholders that let RAGAS import cleanly.

Call :func:`install` *before* importing ``ragas``.
"""

from __future__ import annotations

import sys
import types


def install() -> None:
    import langchain_community.llms as _llms

    if not hasattr(_llms, "VertexAI"):

        class VertexAI:  # placeholder; never instantiated for local eval
            pass

        _llms.VertexAI = VertexAI

    name = "langchain_community.chat_models.vertexai"
    if name not in sys.modules:
        module = types.ModuleType(name)

        class ChatVertexAI:  # placeholder; never instantiated for local eval
            pass

        module.ChatVertexAI = ChatVertexAI
        sys.modules[name] = module
