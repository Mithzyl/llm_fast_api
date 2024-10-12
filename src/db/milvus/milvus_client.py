import os
from typing import List, Any

from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from utils.util import read_yaml_config


class MilvusClient(Milvus):
    def __init__(self, uri: str, embedding: OpenAIEmbeddings):
        super().__init__(embedding_function=embedding,
                         connection_args={"uri": uri})
        self.vector_store = None

    def save_to_vector(self, documents: List[Document] | Any) -> None:
        self.vector_store = self.add_documents(documents)

    def search_similarity(self, query: str, k: int) -> list[Document]:
        return self.similarity_search(query=query, k=k)


def connect_to_milvus():
    db_config = read_yaml_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"))[
        'milvus_config']
    return MilvusClient(uri=db_config['uri'], embedding=OpenAIEmbeddings(model=db_config['model']))
