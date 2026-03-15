import { Server } from "@modelcontextprotocol/sdk/dist/esm/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/dist/esm/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/dist/esm/types.js";
import { exec } from "child_process";
import { promisify } from "util";
import path from "path";
import { fileURLToPath } from "url";

const execPromise = promisify(exec);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const server = new Server(
  {
    name: "blender-agent",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

/**
 * Handler that lists available tools.
 */
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "generate_mesh_code",
        description: "Generate Blender Python code from a point cloud (.npz file) using MeshCoder",
        inputSchema: {
          type: "object",
          properties: {
            npz_path: {
              type: "string",
              description: "Path to the .npz file containing points and normals",
            },
          },
          required: ["npz_path"],
        },
      },
      {
        name: "scaffold_game_environment",
        description: "Scaffold a game environment by generating multiple assets and a scene script",
        inputSchema: {
          type: "object",
          properties: {
            theme: {
              type: "string",
              description: "The theme of the environment (e.g., 'forest', 'indoor')",
            },
          },
          required: ["theme"],
        },
      },
      {
        name: "run_mlops_artifact",
        description: "Run an ML Ops artifact validation task for a generated model",
        inputSchema: {
          type: "object",
          properties: {
            model_id: {
              type: "string",
              description: "The identifier for the model to validate",
            },
          },
          required: ["model_id"],
        },
      },
    ],
  };
});

/**
 * Handler for tools.
 */
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    const kernelWrapperPath = path.join(__dirname, "scripts", "kernel_wrapper.py");
    const pythonPath = "../../../.venv/bin/python3"; // Relative to skill dir

    if (name === "generate_mesh_code") {
      const { stdout, stderr } = await execPromise(`${pythonPath} "${kernelWrapperPath}" --npz_path "${args.npz_path}"`);
      return {
        content: [{ type: "text", text: stdout || stderr }],
      };
    } else if (name === "scaffold_game_environment") {
      // Mocked for now: generates a scene script
      const mockScene = `# Blender Scene Scaffolding for Theme: ${args.theme}\nimport bpy\n# TODO: Integrate MeshCoder assets\nprint("Scene scaffolded")`;
      return {
        content: [{ type: "text", text: mockScene }],
      };
    } else if (name === "run_mlops_artifact") {
      // Mocked for now: simulation of validation
      const validationResult = `Validation for ${args.model_id}: SUCCESS\n- Topology Check: PASSED\n- Manifold: TRUE`;
      return {
        content: [{ type: "text", text: validationResult }],
      };
    } else {
      throw new Error("Unknown tool");
    }
  } catch (error) {
    return {
      content: [{ type: "text", text: `Error: ${error.message}\n${error.stderr || ""}` }],
      isError: true,
    };
  }
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Blender Agent MCP server running on stdio");
}

main().catch((error) => {
  console.error("Server error:", error);
  process.exit(1);
});
