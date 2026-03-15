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
    name: "api-auditor",
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
 * Exposes an "audit_url" tool that checks the status of a given URL.
 */
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "audit_url",
        description: "Audit and test an API endpoint URL",
        inputSchema: {
          type: "object",
          properties: {
            url: {
              type: "string",
              description: "The URL to audit (e.g., https://api.example.com)",
            },
          },
          required: ["url"],
        },
      },
    ],
  };
});

/**
 * Handler for the audit_url tool.
 * Uses the existing scripts/audit.js logic.
 */
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name !== "audit_url") {
    throw new Error("Unknown tool");
  }

  const url = request.params.arguments?.url;
  if (!url) {
    throw new Error("URL is required");
  }

  try {
    const auditScriptPath = path.join(__dirname, "scripts", "audit.js");
    const { stdout, stderr } = await execPromise(`node "${auditScriptPath}" "${url}"`);
    
    return {
      content: [
        {
          type: "text",
          text: stdout || stderr,
        },
      ],
    };
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: `Audit failed: ${error.message}${error.stderr ? `\n${error.stderr}` : ""}`,
        },
      ],
      isError: true,
    };
  }
});

/**
 * Start the server using stdio transport.
 * This allows the server to communicate with any MCP client.
 */
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("API Auditor MCP server running on stdio");
}

main().catch((error) => {
  console.error("Server error:", error);
  process.exit(1);
});
