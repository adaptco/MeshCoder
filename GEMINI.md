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
