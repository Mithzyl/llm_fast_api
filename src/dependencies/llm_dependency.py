import os
from llm.llm_api import LlmApi
from llm.mcp.mcp_tool_manager import MCPToolManager
from llm.state.llm_state import LlmGraph

# --- Global Singleton Instance ---
# This ensures that the entire application shares a single instance of the LlmGraph.
_llm_graph_instance = None

def _initialize_llm_graph():
    """Initializes the LlmGraph instance."""
    global _llm_graph_instance
    if _llm_graph_instance is None:
        print("Initializing LlmGraph instance...")
        # Configure MCP client
        mcp_server_configs = {
            "agent_tools": {
                "command": "python",
                "args": [os.path.abspath("src/llm/mcp/mcp_tools.py")],
                "transport": "stdio",
            },
            "llm_tools": {
                "command": "python",
                "args": [os.path.abspath("src/llm/mcp/generation_tools.py")],
                "transport": "stdio",
            }
        }
        mcp_tool_manager = MCPToolManager(mcp_server_configs)
        # Note: In a long-running app, you might await this, but for dependency injection,
        # we often rely on the main app's lifespan event to handle async setup.
        # For simplicity here, we assume it's handled or can be loaded synchronously for setup.
        
        llm_api = LlmApi(model="deepseek-chat", temperature=0.1, mcp_tool_manager=mcp_tool_manager)
        _llm_graph_instance = LlmGraph(llm_api)
        print("LlmGraph instance initialized.")
    return _llm_graph_instance

def get_llm_graph() -> LlmGraph:
    """
    Dependency injector that provides a singleton LlmGraph instance.
    """
    return _initialize_llm_graph()

# Initialize it once on module load.
_initialize_llm_graph()
