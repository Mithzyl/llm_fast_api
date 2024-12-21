import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


class CustomFileLoader:
    """
    Manages different lang chain loader class
    """
    loader: GenericLoader
    splitter: RecursiveCharacterTextSplitter

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=768,
            chunk_overlap=20,
        )


    def load_pdf_files(self, folder_path: str):
        files = os.listdir(folder_path)
        documents = []
        for file in files[:5]:
            if file.endswith(".pdf"):
                print(f"loading {file}\n")
                pages = PyPDFLoader(os.path.join(folder_path, file))
                documents.append(pages.load_and_split(self.splitter))

        return documents


if __name__ == "__main__":
    loader = CustomFileLoader()
    loader.load_pdf_files("/Users/mith/Desktop/project/llm_fast_api/src/files/paper")
