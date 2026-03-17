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
import fs from "fs/promises";

const execPromise = promisify(exec);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const server = new Server(
  {
    name: "parquet-rag",
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
        name: "docling_to_parquet",
        description: "Convert documents to Parquet chunks using Docling logic",
        inputSchema: {
          type: "object",
          properties: {
            input_path: { type: "string", description: "Path to the input document" },
            output_parquet: { type: "string", description: "Path where the parquet file should be saved" }
          },
          required: ["input_path", "output_parquet"],
        },
      },
      {
        name: "query_multimodal_rag",
        description: "Query the Multimodal RAG stored in Parquet files",
        inputSchema: {
          type: "object",
          properties: {
            query: { type: "string", description: "The natural language query" },
            parquet_path: { type: "string", description: "Path to the parquet knowledge base" }
          },
          required: ["query", "parquet_path"],
        },
      },
      {
        name: "reason_with_dmn",
        description: "Execute DMN logic to decide on translation/arbitration steps",
        inputSchema: {
          type: "object",
          properties: {
            context: { type: "object", description: "The facts and state for DMN arbitration" }
          },
          required: ["context"],
        },
      }
    ],
  };
});

/**
 * Handler for tools.
 */
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    const handlerPath = path.join(__dirname, "scripts", "parquet_handler.py");
    const pythonPath = "../../../.venv/bin/python3";

    if (name === "docling_to_parquet") {
      const { stdout, stderr } = await execPromise(`${pythonPath} "${handlerPath}" --action convert --input "${args.input_path}" --output "${args.output_parquet}"`);
      return { content: [{ type: "text", text: stdout || stderr }] };
    } else if (name === "query_multimodal_rag") {
      const { stdout, stderr } = await execPromise(`${pythonPath} "${handlerPath}" --action query --query "${args.query}" --path "${args.parquet_path}"`);
      return { content: [{ type: "text", text: stdout || stderr }] };
    } else if (name === "reason_with_dmn") {
      const { stdout, stderr } = await execPromise(`${pythonPath} "${handlerPath}" --action dmn --context '${JSON.stringify(args.context)}'`);
      return { content: [{ type: "text", text: stdout || stderr }] };
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
  console.error("Parquet-RAG MCP server running on stdio");
}

main().catch((error) => {
  console.error("Server error:", error);
  process.exit(1);
});
