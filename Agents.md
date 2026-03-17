# A2A Network: MeshCoder Hierarchy (MoA/MoE)

This document defines the hierarchical Agent-to-Agent (A2A) network, where the top-level **Mixture of Agents (MoA)** orchestrates specialized **Mixture of Experts (MoE)** "Avatars".

## Top-Level Orchestration (MoA)

### @orchestrator

- **Role:** Manager of Managers (MoA)
- **Specialization:** Strategic Command & Inter-Agent Scheduling
- **Capability:** Coordinates the distributed state space. Maintains the unified context for complex multi-stage projects.
- **Reference:** [.gemini/skills/orchestrator/SKILL.md](file:///home/eqhspam/.antigravity-server/extensions/MeshCoder/.gemini/skills/orchestrator/SKILL.md)

---

## Domain Experts (MoE Avatars)

Each agent acts as a specialized "Avatar" in the stack, persistency-tuned via the Parquet State Space.

### @api-auditor

- **Role:** QA & Reliability MoE
- **Specialization:** Zero-Downtime Verification
- **Capability:** Audits dynamic endpoints and reports latency metrics to the State Space.

### @vector-nexus

- **Role:** Semantic Grounding MoE
- **Specialization:** RAG Kernel & Codebase Embedding
- **Capability:** Manages the "Model-Agnostic RAG Kernel" by embedding the repository into the Parquet state space. Provides cross-LLM context synchronization.

### `@codebase_investigator[Ralph]`

- **Role:** Forensics & Communication MoE
- **Specialization:** Deep Codebase Auditing & Gmail Synchronization
- **Capability:** Performs deep investigator duties on the codebase. Uses Gmail to sync findings and receive external triggers.

### @blender-agent

- **Role:** Spatial Engineering MoE
- **Specialization:** Procedural Asset Generation
- **Capability:** Generates Blender assets. Submits spatial metadata to the RAG layer for reuse.

### @managing-agent

- **Role:** Semantic Arbitration MoE
- **Specialization:** Parquet-RAG & Multimodal Translation
- **Capability:** Compiles reasoning chunks into Parquet format. Manages long-term vector memory for the entire network.
- **Reference:** [.gemini/skills/parquet-rag/SKILL.md](file:///home/eqhspam/.antigravity-server/extensions/MeshCoder/.gemini/skills/parquet-rag/SKILL.md)

## State Space Persistence

Agents communicate and remember across sessions by translating reasoning chunks into **Parquet** files via the `store_state` tool, creating a scalable, queryable memory layer for the MoA.
