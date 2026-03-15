---
description: How to coordinate tasks across multiple agents using the Orchestrator
---

# Orchestration Workflow

Follow these steps when managing tasks across multiple specialized agents in the A2A network.

1.  **Analyze Objective**: The Orchestrating Agent receive a high-level goal from the user.
2.  **Define Tasks**: Decompose the goal into specific, non-overlapping tasks.
3.  **Assign Agents**:
    - Identify the best specialized agent for each task (e.g., `@api-auditor` for reliability).
    - Update [A2A_TASKS.md](file:///home/eqhspam/.antigravity-server/extensions/MeshCoder/A2A_TASKS.md) with the assignment and status.
4.  **Monitor Progress**:
    - Check the `task.md` of each agent (if accessible) or their updates in `A2A_TASKS.md`.
    - Handle blockers or clarify requirements between agents.
5.  **Consolidate Results**:
    - Once all sub-tasks are complete, the Orchestrator verifies the integrated solution.
    - Update [walkthrough.md](file:///home/eqhspam/.gemini/antigravity/brain/9e6690ed-274a-45dd-9f4d-29c96622543b/walkthrough.md) with the combined outcomes.
6.  **Finalize**: Notify the user that the orchestrated task is complete.
