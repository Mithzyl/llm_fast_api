from llm.mem0.mem0_client import CustomMemoryClient, mem0_config


def get_memory_client() -> CustomMemoryClient:
    config = mem0_config
    return CustomMemoryClient(config)
