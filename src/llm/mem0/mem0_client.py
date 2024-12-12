import os
from mem0 import Memory
config = {
    "vector_store": {
        "provider": "milvus",
        "config": {
            "collection_name": "quickstart_mem0_with_milvus",
            "embedding_model_dims": "768",
            "url": "http://localhost:19530",  # Use local vector database for demo purpose
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini-2024-07-18",
            "temperature": 0.2,
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

# Initialize Memory with the configuration
m = Memory.from_config(config)

# Add a memory
m.add("I'm visiting Paris", user_id="john")

# Retrieve memories
memories = m.get_all(user_id="john")
print(memories)


