from typing import List, Any

from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from milvus_model.base import RerankResult
from milvus_model.reranker.bgereranker import BGERerankFunction

from utils.util import read_yaml_config


class CustomMilvusClient(Milvus):
    def __init__(self, uri: str, embedding_model: Any, collection_name: str):

        self.embedding_model = embedding_model
        super().__init__(embedding_function=self.embedding_model,
                         connection_args={"uri": uri},
                         collection_name=collection_name,
                         drop_old=False)
        self.auto_id = True
        self.reranker = BGERerankFunction(
                            model_name="BAAI/bge-reranker-base",  # Specify the model name. Defaults to `BAAI/bge-reranker-v2-m3`.
                            device="cpu" # Specify the device to use, e.g., 'cpu' or 'cuda:0'
                        )


    def save_to_vector(self, documents: List[List[Document]] | Any) -> None:
        for document in documents:
            self.add_documents(document)

    def search_similarity(self, query: str, k: int) -> list[Document]:
        return self.similarity_search(query=query, k=k)

    def rerank_document(self, documents: List[str], query: str) -> list[RerankResult]:
        result = self.reranker(query=query, documents=documents, top_k=5)

        return result



def connect_to_milvus():
    db_config = read_yaml_config(r"/Users/mith/Desktop/project/llm_fast_api/src/config.yaml")['milvus_config']

    # return CustomMilvusClient(uri=db_config['uri'],
    #                           embedding_model=OpenAIEmbeddings(model=db_config['model'],
    #                           api_key=db_config['api_key']),
    #                           collection_name='IKEA')

    return CustomMilvusClient(uri=db_config['uri'],
                              embedding_model=OllamaEmbeddings(model="nomic-embed-text:latest"),
                              collection_name='IKEA')

