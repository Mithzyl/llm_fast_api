from typing import List, Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings

from utils.util import read_yaml_config


class CustomMilvusClient(Milvus):
    def __init__(self, uri: str, embedding_model: Any, collection_name: str):

        self.embedding_model = embedding_model
        super().__init__(embedding_function=self.embedding_model,
                         connection_args={"uri": uri},
                         collection_name=collection_name,
                         drop_old=True)
        self.auto_id = True



    def save_to_vector(self, documents: List[Document] | Any) -> None:
        self.add_documents(documents)

    def search_similarity(self, query: str, k: int) -> list[Document]:
        return self.similarity_search(query=query, k=k)


def connect_to_milvus():
    db_config = read_yaml_config(r"D:\project\python\fastapi\fastApiDemo\src\config.yaml")['milvus_config']

    return CustomMilvusClient(uri=db_config['uri'],
                              embedding_model=OpenAIEmbeddings(model=db_config['model'],
                              api_key=db_config['api_key']),
                              collection_name=db_config['collection_name'])

