import os

from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter

from db.milvus.milvus_client import connect_to_milvus


class CustomFileLoader:
    """
    Manages different lang chain loader class
    """
    loader: GenericLoader
    text_splitter: RecursiveCharacterTextSplitter
    json_splitter: RecursiveJsonSplitter

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1536,
            chunk_overlap=100,
        )
        self.json_splitter = RecursiveJsonSplitter()


    def load_pdf_files(self, folder_path: str):
        files = os.listdir(folder_path)
        documents = []
        for file in files:
            if file.endswith(".pdf"):
                print(f"Loading {file}")
                pages = PyPDFLoader(os.path.join(folder_path, file))
                documents.append(pages.load_and_split(self.text_splitter))

        return documents

    def load_json_files(self, folder_path: str):
        files = os.listdir(folder_path)
        documents = []
        for file in files:
            if file.endswith(".json"):
                print(f"Loading {file}")
                pages = JSONLoader(os.path.join(folder_path, file),
                                   ".[].messages[]",
                                   ".text",
                                   is_content_key_jq_parsable=True,
                                   text_content=False)
                documents.append(pages.load())

        return documents

    def store(self, config: str, folder_path: str):
        documents = self.load_json_files(folder_path)
        vector_store = connect_to_milvus()
        vector_store.save_to_vector(documents)


if __name__ == "__main__":
    loader = CustomFileLoader()
    loader.store("", "/Users/mith/Desktop/project/llm_fast_api/src/files/records")
