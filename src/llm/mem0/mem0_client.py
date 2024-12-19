import os
from typing import Dict, Any

from mem0 import Memory, MemoryClient
from openai import api_version

from utils.util import set_api_key_environ

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

    def get_all_memory_by_conversation_id(self, conversation_id) -> dict:
        try:
            memories = self.memory.get_all(run_id=conversation_id)
            return memories
        except Exception as e:
            return None


    def search_memory_by_user_id(self, message: str, user_id: str) -> dict:
        """
        search relevant memory of a user by user_id
        """
        try:
            memories = self.memory.search(query=message, user_id=user_id)
            return memories
        except Exception as e:
            return None

    def search_memory_by_conversation_id(self, message: str, conversation_id: str) -> dict:
        """
        search relevant memory of a conversation by conversation id
        """
        try:
            memories = self.memory.search(query=message, run_id=conversation_id)
            return memories
        except Exception as e:
            return None

    def add_memory_by_conversation_id(self, messages: str, conversation_id: str) -> dict:
        """
        Add or update the memory of a conversation given its session_id
        The goal is to infer and remember key information as the on going conversation
        E.g.
        1. Before generating a response, let the model recall the memory
        2. After the generation, update the memory
        Args:
            messages: The messages (can include history) of the conversation
            conversation_id: the id of the conversation session

        Returns:

        """
        memory = self.memory.add(messages, run_id=conversation_id)
        return memory



if __name__ == "__main__":
    set_api_key_environ("../../key.json")
    # Initialize Memory with the configuration
    m = MemoryClient(api_key="")


    # Add a memory
    m.add("My name is Mith, My old home was designed by myself",
                               run_id="123e4567-e89b-12d3-a456-426614174000")

    # Retrieve memories
    # memories = m.get_all(run_id="123e4567-e89b-12d3-a456-426614174000")
    # print(memories)


