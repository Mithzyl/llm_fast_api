from typing import Dict, Any, List

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPToolManager:
    """
    A wrapper class for managing connections to one or more MCP servers.

    text
    This class simplifies the configuration, tool loading, and lifecycle management of MultiServerMCPClient.
    It is recommended to use the asynchronous context manager (`async with`) to ensure connections are properly closed.

    Usage:
        configs = { ... }
        async with MCPToolManager(configs) as tool_manager:
            tools = tool_manager.tools
            tool_map = tool_manager.tool_map
            # ... Use tools and tool_map ...
    """
    def __init__(self, server_configs: Dict[str, Any]):
        """
        Initializes a new instance of MCPToolManager.
        Args:
            server_configs: A dict containing one or multiple MCP server configurations.
        """
        if not server_configs:
            raise ValueError("MCPToolManager requires at least one MCP server configuration")
        self.server_configs = server_configs
        self._client: MultiServerMCPClient | None = None
        self._tools: List[BaseTool] = []
        self._tool_map: Dict[str, BaseTool] = {}

    async def load(self) -> None:
        """
        Loads all configured MCP servers.
        Returns:

        """
        self._client = MultiServerMCPClient(self.server_configs)
        try:
            self._tools = await self._client.get_tools()
            self._tool_map = {tool.name: tool for tool in self._tools}
            print(f"成功加载 {len(self._tools)} 个工具: {[tool.name for tool in self._tools]}") # TODO: Replace with logging
        except Exception as e:
            print(f"从 MCP 服务器加载工具失败: {e}")

            raise

    async def close(self) -> None:
        """
        Turn off all MCP server connections.
        Returns:

        """
        pass


    @property
    def tools(self) -> List[BaseTool]:
        return self._tools

    @property
    def tool_map(self) -> Dict[str, BaseTool]:
        return self._tool_map

    async def __aenter__(self):
        """
        async context manager entry
        Returns:

        """
        await self.load()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
