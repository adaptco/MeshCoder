# Blender Agent Skill

A skill for generating 3D assets and environments using MeshCoder, integrated into an ML Ops workflow for AI game development.

## Description

The Blender Agent acts as a bridge between high-level architectural requirements and low-level asset generation. It leverages the MeshCoder algorithm to reconstruct 3D meshes from point clouds and provides tools for scaffolding complex game environments.

## Tools

### generate_mesh_code
Generates a Blender Python script from a point cloud (.npz file).
- **Arguments:**
  - `npz_path`: Path to the .npz data file.

### scaffold_game_environment
Creates a high-level scene script combining multiple generated assets.
- **Arguments:**
  - `theme`: Theme of the environment (e.g., "dungeon", "meadow").

### run_mlops_artifact
Validates a generated model as part of an ML Ops pipeline.
- **Arguments:**
  - `model_id`: ID of the model to validate.

## Implementation Details

The skill uses an MCP server (`mcp_server.js`) to expose these tools. The actual inference is handled by `scripts/kernel_wrapper.py` which interfaces with the MeshCoder algorithm.
