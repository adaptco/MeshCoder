---
name: mesh-coder-host
description: Unified MCP Host for large dataset orchestration and web-augmented RAG.
---

# MeshCoder Host Skill

This skill provides a unified gateway to the dual-language MCP architecture. It enables high-performance web search, Parquet-based state persistence, and hierarchical agent orchestration.

## Tools

- `search_web`: Multi-engine web search with session affinity.
- `fetch_page`: Clean content extraction from URLs.
- `store_state`: Persistent Parquet storage for agent reasoning and state spaces.
- `query_vector`: RAG-based retrieval of agent memory and context.

## Architecture

- **Gateway (TypeScript)**: Handles resilience (circuit breakers, retries) and routing.
- **Backend (Python)**: Specialized domain logic for search and data processing.
- **Persistence**: Parquet-based state space for long-term agent memory.
