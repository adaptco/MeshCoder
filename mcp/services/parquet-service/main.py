from fastmcp import FastMCP
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from embeddings import get_kernel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("parquet-service")

app = FastMCP("parquet-service")

# Base directory for parquet storage
BASE_DATA_DIR = Path(__file__).parent / "data" / "parquet"
BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

@app.tool()
def store_state(agent_id: str, state_data: dict, namespace: str = "default") -> dict:
    """
    Save agent state or reasoning chunks to Parquet for orchestration persistence.
    """
    logger.info(f"Storing state for agent: {agent_id} in namespace: {namespace}")
    
    # Generate embedding for the data if it's text-like
    text_content = json.dumps(state_data)
    embedding = get_kernel().generate_embeddings([text_content])[0]
    
    record = {
        "agent_id": agent_id,
        "timestamp": datetime.now().isoformat(),
        "namespace": namespace,
        "data_json": text_content,
        "embedding": embedding.tolist()
    }
    
    df = pd.DataFrame([record])
    file_path = BASE_DATA_DIR / f"{namespace}.parquet"
    
    try:
        if file_path.exists():
            existing_df = pd.read_parquet(file_path)
            df = pd.concat([existing_df, df], ignore_index=True)
            
        df.to_parquet(file_path, engine='pyarrow', index=False)
        return {"status": "success", "file_path": str(file_path), "rows": len(df)}
    except Exception as e:
        logger.error(f"Failed to save parquet: {e}")
        return {"status": "error", "message": str(e)}

@app.tool()
def ingest_codebase(root_path: str, domain: str = "codebase") -> dict:
    """
    Recursively scan and embed the codebase into the Parquet state space for RAG.
    """
    logger.info(f"Ingesting codebase from: {root_path}")
    docs = []
    
    for path in Path(root_path).rglob("*"):
        if path.is_file() and not any(part.startswith(".") for part in path.parts):
            if path.suffix in [".py", ".ts", ".js", ".md", ".yaml", ".json"]:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if content.strip():
                            docs.append({"path": str(path), "content": content})
                except Exception as e:
                    logger.warning(f"Could not read {path}: {e}")

    if not docs:
        return {"status": "success", "message": "No indexable files found."}

    # Embed and store
    texts = [f"File: {d['path']}\nContent: {d['content'][:1000]}" for d in docs]
    embeddings = get_kernel().generate_embeddings(texts)
    
    records = []
    for i, doc in enumerate(docs):
        records.append({
            "agent_id": "vector-nexus",
            "timestamp": datetime.now().isoformat(),
            "namespace": domain,
            "data_json": json.dumps(doc),
            "embedding": embeddings[i].tolist()
        })
        
    df = pd.DataFrame(records)
    file_path = BASE_DATA_DIR / f"{domain}.parquet"
    df.to_parquet(file_path, engine='pyarrow', index=False)
    
    return {"status": "success", "files_ingested": len(docs), "domain": domain}

@app.tool()
def query_vector(query: str, top_k: int = 5, domain: str = "default") -> dict:
    """
    Query the RAG system for context using vector similarity.
    """
    logger.info(f"Querying vector space for: {query} in domain: {domain}")
    
    file_path = BASE_DATA_DIR / f"{domain}.parquet"
    
    if not file_path.exists():
        return {"results": [], "message": f"No data found for domain: {domain}"}
    
    try:
        df = pd.read_parquet(file_path)
        if "embedding" not in df.columns:
            return {"results": [], "message": "No vector data found in this domain."}

        # Vector search
        query_embedding = get_kernel().generate_embeddings([query])[0]
        doc_embeddings = np.array(df["embedding"].tolist())
        
        similarities = get_kernel().compute_similarity(query_embedding, doc_embeddings)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            row = df.iloc[idx]
            results.append({
                "agent_id": row["agent_id"],
                "timestamp": row["timestamp"],
                "content": json.loads(row["data_json"]),
                "similarity": float(similarities[idx])
            })
            
        return {"results": results}
    except Exception as e:
        logger.error(f"Failed to query parquet: {e}")
        return {"status": "error", "message": str(e)}

@app.tool()
def generate_spatial_tensor(text_queries: list[str], domain: str = "default") -> dict:
    """
    Map a list of text queries to 3D tensors for CAD environment embodiment.
    """
    logger.info(f"Generating spatial tensors for {len(text_queries)} queries in domain: {domain}")
    
    try:
        # 1. Get high-dim embeddings
        embeddings = get_kernel().generate_embeddings(text_queries)
        
        # 2. Map to 3D via Spatial Ingester
        tensors_3d = get_spatial_ingester().map_to_3d(embeddings)
        
        # 3. Format for Blender consumption
        blender_data = format_for_blender(tensors_3d)
        
        return {
            "status": "success",
            "tensors": blender_data,
            "domain": domain
        }
    except Exception as e:
        logger.error(f"Failed to generate spatial tensors: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    app.run()
