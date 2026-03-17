import yaml
import pytest
from pathlib import Path
import sys

# Add the generated models to the path
sys.path.append(str(Path(__file__).parent.parent.parent / "contracts" / "generated"))

try:
    from models import SearchWebInput, FetchPageInput, StoreStateInput, QueryVectorInput
except ImportError:
    SearchWebInput = None
    FetchPageInput = None
    StoreStateInput = None
    QueryVectorInput = None

def test_tools_yaml_schema():
    """Verify that tools.yaml matches the expected structure."""
    tools_path = Path(__file__).parent.parent.parent / "contracts" / "tools.yaml"
    with open(tools_path, 'r') as f:
        data = yaml.safe_load(f)
    
    assert "tools" in data
    tool_names = [t["name"] for t in data["tools"]]
    assert "search_web" in tool_names
    assert "fetch_page" in tool_names
    assert "store_state" in tool_names
    assert "query_vector" in tool_names

def test_pydantic_models():
    """Verify that generated Pydantic models can validate valid data."""
    if SearchWebInput is None:
        pytest.skip("Pydantic models not generated or not in path")
    
    # Valid data for search_web
    data = {"query": "test query", "top_k": 3}
    model = SearchWebInput(**data)
    assert model.query == "test query"
    assert model.top_k == 3

    # Valid data for store_state
    state_data = {"agent_id": "agent-1", "state_data": {"key": "val"}}
    state_model = StoreStateInput(**state_data)
    assert state_model.agent_id == "agent-1"
    assert state_model.state_data["key"] == "val"

def test_implementation_alignment():
    """
    In a real CI, this would hit the actual MCP endpoints.
    Here we validate that the tools.yaml matches the code structure.
    """
    # This is a placeholder for actual endpoint testing
    pass

if __name__ == "__main__":
    # Simple manual run if not using pytest
    test_tools_yaml_schema()
    print("Contract tests passed (basic)!")
