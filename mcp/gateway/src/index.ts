import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import axios from "axios";
import { ServiceRegistry } from "./service-registry.js";

const serviceRegistry = new ServiceRegistry({
  search_web: ["http://localhost:8000"],
  fetch_page: ["http://localhost:8000"],
  open_pr: ["http://localhost:8001"],
  store_state: ["http://localhost:8002"],
  query_vector: ["http://localhost:8002"],
  ingest_codebase: ["http://localhost:8002"],
  generate_spatial_tensor: ["http://localhost:8002"],
  list_emails: ["http://localhost:8003"],
  read_email: ["http://localhost:8003"],
  send_email: ["http://localhost:8003"]
});

const server = new Server(
  {
    name: "ts-mcp-gateway",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "search_web",
        description: "Search the web for real-time information with multi-engine fallback.",
        inputSchema: {
          type: "object",
          properties: {
            query: { type: "string" },
            top_k: { type: "integer", default: 5 }
          },
          required: ["query"]
        },
      },
      {
        name: "fetch_page",
        description: "Extract clean content from a URL using Firecrawl/Playwright.",
        inputSchema: {
          type: "object",
          properties: {
            url: { type: "string" }
          },
          required: ["url"]
        },
      },
      {
        name: "store_state",
        description: "Save agent state or reasoning chunks to Parquet for orchestration persistence.",
        inputSchema: {
          type: "object",
          properties: {
            agent_id: { type: "string" },
            state_data: { type: "object" },
            namespace: { type: "string", default: "default" }
          },
          required: ["agent_id", "state_data"]
        },
      },
      {
        name: "query_vector",
        description: "Query the RAG system for context using vector metadata stored in Parquet/VectorStore.",
        inputSchema: {
          type: "object",
          properties: {
            query: { type: "string" },
            top_k: { type: "integer", default: 5 },
            domain: { type: "string" }
          },
          required: ["query"]
        },
      },
      {
        name: "ingest_codebase",
        description: "Recursively scan and embed the codebase into the Parquet state space for RAG-based context.",
        inputSchema: {
          type: "object",
          properties: {
            root_path: { type: "string" },
            domain: { type: "string", default: "codebase" }
          },
          required: ["root_path"]
        },
      },
      {
        name: "list_emails",
        description: "List recent emails from Gmail using OAuth.",
        inputSchema: {
          type: "object",
          properties: {
            max_results: { type: "integer", default: 10 },
            query: { type: "string" }
          }
        },
      },
      {
        name: "read_email",
        description: "Read the content of a specific email by ID.",
        inputSchema: {
          type: "object",
          properties: {
            email_id: { type: "string" }
          },
          required: ["email_id"]
        },
      },
      {
        name: "send_email",
        description: "Send an email via Gmail.",
        inputSchema: {
          type: "object",
          properties: {
            to: { type: "string" },
            subject: { type: "string" },
            body: { type: "string" }
          },
          required: ["to", "subject", "body"]
        },
      },
      {
        name: "generate_spatial_tensor",
        description: "Map a list of text queries to 3D tensors for CAD environment embodiment.",
        inputSchema: {
          type: "object",
          properties: {
            text_queries: {
              type: "array",
              items: { type: "string" }
            },
            domain: { type: "string", default: "default" }
          },
          required: ["text_queries"]
        },
      }
    ],
  };
});

import CircuitBreaker from "opossum";
import axiosRetry from "axios-retry";

// Configure retry
axiosRetry(axios, { retries: 3, retryDelay: axiosRetry.exponentialDelay });

const breakerOptions = {
  timeout: 5000, // 5 seconds
  errorThresholdPercentage: 50,
  resetTimeout: 10000 // 10 seconds
};

async function callService(service: any, name: string, args: any) {
  const response = await axios.post(`${service.baseUrl}/call_tool`, {
    name,
    arguments: args
  });
  return response.data;
}

const breakers: Map<string, CircuitBreaker> = new Map();

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const services = serviceRegistry.get(name);

  if (services.length === 0) {
    throw new Error(`Tool ${name} not found`);
  }

  const primary = services[0];
  const fallback = services[1];

  let breaker = breakers.get(primary.baseUrl);
  if (!breaker) {
    breaker = new CircuitBreaker(callService, breakerOptions);
    breakers.set(primary.baseUrl, breaker);
  }

  try {
    return await breaker.fire(primary, name, args);
  } catch (error) {
    console.warn(`Primary failed for ${name}, trying fallback...`);
    if (fallback) {
      try {
        return await callService(fallback, name, args);
      } catch (fallbackError) {
        console.error(`Fallback failed for ${name}`);
        throw fallbackError;
      }
    }
    throw error;
  }
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("MCP TS Gateway running on stdio");
}

main().catch((error) => {
  console.error("Fatal error in main():", error);
  process.exit(1);
});
