import os
from typing import Dict, Any

from mem0 import Memory, MemoryClient
from openai import api_version

mem0_config = {
    "vector_store": {
        "provider": "milvus",
        "config": {
            "collection_name": "mem0",
            "embedding_model_dims": "768",
            "url": "http://localhost:19530",  # Use local vector database for demo purpose
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini-2024-07-18",
            "temperature": 0.9,
            "max_tokens": 1500,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            # Alternatively, you can use "snowflake-arctic-embed:latest"
            "ollama_base_url": "localhost:11434",
        },
    },
}



class CustomMemoryClient:
    def __init__(self, config):

        self.memory = Memory.from_config(mem0_config)

    def add_memory_by_user_id(self, message: str, user_id: str) -> dict[str, Any]:
        # TODO: Integrate into langgraph (get message and user_id from state)
        memory = self.memory.add(message, user_id=user_id)
        return memory

    def get_all_memory_by_user_id(self, user_id) -> dict:
        try:
            memories = self.memory.get_all(user_id=user_id)
            return memories
        except Exception as e:
            return None

    def search_memory_by_user_id(self, message: str, user_id: str) -> dict:
        """
        search relevant memory of a conversation by user_id
        """
        try:
            memories = self.memory.search(query=message, user_id=user_id)
            return memories
        except Exception as e:
            return None



if __name__ == "__main__":
    # Initialize Memory with the configuration
    m = Memory.from_config(mem0_config)
    # Add a memory
    m.add("I'm visiting Paris", user_id="john")

    # Retrieve memories
    memories = m.get_all(user_id="john")
    print(memories)


