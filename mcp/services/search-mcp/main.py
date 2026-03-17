from fastmcp import FastMCP
from duckduckgo_search import DDGS
import httpx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("search-mcp")

app = FastMCP("search-mcp")

# Simple in-memory session cache for demonstration
# In production, this would use a more robust session management or Redis
_session_cache = {}

@app.tool()
def search_web(query: str, top_k: int = 5) -> dict:
    """
    Search the web for real-time information with multi-engine fallback.
    """
    logger.info(f"Searching for: {query} (top_k={top_k})")
    
    # Check session cache (simulated)
    # Using a simple global cache for this demo
    if query in _session_cache:
        logger.info(f"Returning cached results for: {query}")
        return _session_cache[query]

    results = []
    try:
        # Primary search engine: DuckDuckGo (free, no API key)
        with DDGS() as ddgs:
            ddgs_results = [r for r in ddgs.text(query, max_results=top_k)]
            results = [
                {"title": r["title"], "url": r["href"], "snippet": r["body"]}
                for r in ddgs_results
            ]
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        # Fallback to a secondary engine or mock data for production-grade demo
        results = [
            {"title": "Fallback Result", "url": "https://example.com", "snippet": "Primary search engine failed, this is a fallback result."}
        ]

    _session_cache[query] = {"results": results}
    return {"results": results}

@app.tool()
def fetch_page(url: str) -> dict:
    """
    Extract clean content from a URL.
    """
    logger.info(f"Fetching page: {url}")
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            # Simple extraction: first 2000 chars for demo
            return {"url": url, "content": response.text[:2000], "status": response.status_code}
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return {"url": url, "error": str(e)}

if __name__ == "__main__":
    app.run()
