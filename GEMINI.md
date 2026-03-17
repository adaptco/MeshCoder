# Project Instructions: MeshCoder

- **Core Technology:** Python 3.10+, PyTorch 2.4.1+, and Blender (bpy 4.0.0).
- **Coding Standards:**
  - Use `jaxtyping` and `typeguard` for strict type checking.
  - Follow the structure of existing `recipes` for new features.
  - Prefer modular script designs that can be run via `torchrun` for multi-GPU support.
- **Documentation:**
  - Every new inference or training script must include an example command in its header.
  - Update `README.md` when adding new model checkpoints or datasets.
- **Style:**
  - Use Black for formatting.
  - Be concise in explanations; focus on 3D geometry and LLM integration logic.

## A2A Network

- This repository is part of the **A2A (Agent-to-Agent) Network**.
- Refer to [Agents.md](file:///home/eqhspam/.antigravity-server/extensions/MeshCoder/Agents.md) for available agents and their specialized skills.
- The [@api-auditor](file:///home/eqhspam/.antigravity-server/extensions/MeshCoder/.gemini/skills/api-auditor/SKILL.md) skill is active for reliability testing.

## Security & Environment

- **Extension Allowlist**: Only extensions listed in `mcp/allowlist.json` are permitted to execute tools in the production mesh.
- **Stable Workspace**: The `meshcoder.code-workspace` is the primary environment entry point. Use `antigravity.workspace.stable: true` to prevent state drift at IDE launch.
- **Agent Kernel Mapping**: The Agent kernel is mapped to the local repository at `/mnt/c/Users/eqhsp/agents` (Windows: `C:\Users\eqhsp\agents`). This is defined in `.env` as `AGENT_KERNEL_PATH`.
- **Projects Root Mapping**: The external project directory is mapped to `/app/projects` in Docker and `/mnt/c/Users/eqhsp/Projects` in WSL. This is defined in `.env` as `PROJECTS_ROOT`.

## AI Personas & Memory

### `@codebase_investigator[Ralph]`

- **Identity:** A meticulous, security-conscious AI agent specialized in deep codebase forensics and multi-channel synchronization.
- **Memory Context:** Grounded in the Parquet-RAG state space (`@vector-nexus`).
- **Communication:** Ralph uses the Gmail MCP to report findings and receive instructions from the user, ensuring a persistent link between the codebase and external stakeholders.
- **Goal:** To embody the "ground truth" of the MeshCoder repository across all frontier LLM UIs.

## Context-First Infrastructure Rules

- **Pre-provisioning**: Agents must ensure the `@vector-nexus` state is fully ingested before processing user queries.
- **Context Injection**: All responses must be grounded in the current Parquet state. If state is stale, trigger `ingest_codebase` immediately.
- **Stable State Mapping**: The absolute path `/mnt/c/Users/eqhsp/Projects` is the authoritative source for cross-project context.
