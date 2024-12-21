from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.generic import GenericLoader


class CustomFileLoader:
    """
    Manages different lang chain loader class
    """
    loader: GenericLoader

    def load_pdf_files(self, folder_path: str):
        pages = []
        loader = GenericLoader.from_filesystem(folder_path, glob="**/*.pdf", show_progress=True)
        documents = loader.load()

        return documents


if __name__ == "__main__":
    loader = CustomFileLoader()
    loader.load_pdf_files("")
