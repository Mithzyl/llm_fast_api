import logging
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
# 导入您已经写好的 Milvus 连接函数
from db.milvus.milvus_client import connect_to_milvus

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def _load_document(file_path: str):
    """根据文件扩展名选择合适的加载器。"""
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".pdf":
        return PyPDFLoader(file_path)
    elif file_extension == ".md":
        return UnstructuredMarkdownLoader(file_path)
    elif file_extension == ".txt":
        return TextLoader(file_path, encoding='utf-8')
    else:
        logging.warning(f"不支持的文件类型: {file_extension}，跳过文件 {file_path}")
        return None


def process_and_embed_document(file_path: str):
    """
    处理新文件的核心函数：加载 -> 分割 -> 存入 Milvus。
    """
    try:
        logging.info(f"开始处理文件: {file_path}")

        # 1. 加载文档
        loader = _load_document(file_path)
        if not loader:
            return
        documents = loader.load()

        # 2. 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 每个分块的最大字符数
            chunk_overlap=200,  # 分块间的重叠字符数
        )
        split_docs = text_splitter.split_documents(documents)

        # 为每个分块添加源文件路径的元数据，这对于后续的删除操作至关重要
        for doc in split_docs:
            doc.metadata["source"] = file_path

        logging.info(f"文件 {os.path.basename(file_path)} 已被分割成 {len(split_docs)} 个块。")

        # 3. 连接 Milvus 并存储
        milvus_client = connect_to_milvus()
        milvus_client.add_documents(split_docs)

        logging.info(f"成功将 {file_path} 的分块存入 Milvus。")

    except Exception as e:
        logging.error(f"处理文件 {file_path} 时发生错误: {e}", exc_info=True)

