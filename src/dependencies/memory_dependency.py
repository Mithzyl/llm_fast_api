from llm.mem0.mem0_client import MemoryClient, mem0_config


def get_memory_client() -> MemoryClient:
    config = mem0_config
    return MemoryClient(config)