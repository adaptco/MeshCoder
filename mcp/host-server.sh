#!/bin/bash

# Kill background processes on exit
trap "exit" INT TERM ERR
trap "kill 0" EXIT

echo "Starting Python MCP Services..."

# Start Search Service
cd mcp/services/search-mcp
python3 main.py > search.log 2>&1 &
SEARCH_PID=$!
echo "Search Service started (PID: $SEARCH_PID) on port 8000"

# Start Parquet Service
cd ../parquet-service
python3 main.py > parquet.log 2>&1 &
PARQUET_PID=$!
echo "Parquet Service started (PID: $PARQUET_PID) on port 8002"

# Wait a moment for services to bind ports
sleep 5

# --- Provision System ---
echo "Auto-provisioning context from agent-infra.yaml..."
# Trigger ingest_codebase via Parquet Service (localhost:8002)
# Here we assume the root of the project is the parent directory
PROJECT_ROOT=$(pwd | sed 's/\/mcp$//')
curl -X POST http://localhost:8002/call_tool \
     -H "Content-Type: application/json" \
     -d "{\"name\": \"ingest_codebase\", \"arguments\": {\"root_path\": \"$PROJECT_ROOT\"}}" > provisioning.log 2>&1
echo "Codebase ingested into Parquet State Space."

# Start TS Gateway on stdio (this will be the main MCP entry point)
echo "Starting TS Gateway on stdio..."
cd ../../gateway
# Run the gateway using ts-node or compiled dist
# Using npx ts-node to run directly from source for the host
npx ts-node src/index.ts
