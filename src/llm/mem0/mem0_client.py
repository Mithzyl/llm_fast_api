import os
from mem0 import Memory
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
            "ollama_base_url": "http://localhost:11434",
        },
    },
}



class MemoryClient:
    def __init__(self, config):
        self.memory = Memory.from_config(config)

    def add_memory_by_user_id(self, message: str, user_id) -> None:
        # TODO: Integrate into langgraph (get message and user_id from state)
        self.memory.add(message, user_id, api_version="v1.1")

    def get_memory_by_user_id(self, user_id) -> dict:
        return self.memory.get(user_id)



if __name__ == "__main__":

    os.environ["OPENAI_API_KEY"] = "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"
    # Initialize Memory with the configuration
    m = Memory.from_config(mem0_config)
    # Add a memory
    m.add("I'm visiting Paris", user_id="john")

    # Retrieve memories
    memories = m.get_all(user_id="john")
    print(memories)


