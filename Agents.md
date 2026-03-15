# A2A Network: MeshCoder Agents

This document defines the Agent-to-Agent (A2A) network capabilities for the MeshCoder project.

## Active Agents

### @orchestrator
- **Role:** Agent Manager / Orchestrating Agent
- **Specialization:** Task Decomposition & Coordination
- **Capability:** Manages the A2A network and tracks tasks across workspaces.
- **Reference:** [.gemini/skills/orchestrator/SKILL.md](file:///home/eqhspam/.antigravity-server/extensions/MeshCoder/.gemini/skills/orchestrator/SKILL.md)

### @api-auditor
- **Role:** QA & Reliability Engineer
- **Specialization:** API Endpoint Auditing
- **Capability:** Validates URL availability, status codes, and latency.
- **Reference:** [.gemini/skills/api-auditor/SKILL.md](file:///home/eqhspam/.antigravity-server/extensions/MeshCoder/.gemini/skills/api-auditor/SKILL.md)

### @blender-agent
- **Role:** 3D Environment Engineer
- **Specialization:** Procedural Asset Generation
- **Capability:** Uses MeshCoder to generate Blender assets and coordinates ML Ops for game environments.
- **Reference:** [.gemini/skills/blender-agent/SKILL.md](file:///home/eqhspam/.antigravity-server/extensions/MeshCoder/.gemini/skills/blender-agent/SKILL.md)

## Interaction Protocol
Agents in this network interact via structured function calls. See individual skill cards for parameter schemas.
