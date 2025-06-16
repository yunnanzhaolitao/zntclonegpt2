from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredFileLoader,
)
# 使用简单字符分割器，替代需要 NLTK 的 RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def _get_loader(file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(file_path))
    if suffix in [".docx", ".doc"]:
        return UnstructuredWordDocumentLoader(str(file_path))
    if suffix in [".xlsx", ".xls"]:
        return UnstructuredExcelLoader(str(file_path))
    if suffix in [".pptx", ".ppt"]:
        return UnstructuredPowerPointLoader(str(file_path))
    return UnstructuredFileLoader(str(file_path))

def ingest_folder(folder: str, openai_api_key: str = None, persist_dir: str | None = None):
    """
    读取 folder 下所有文件，切分后用本地模型做 embeddings，
    持久化到 persist_dir 目录（Chroma）。
    """
    # 1. 文档加载
    docs = []
    for fp in Path(folder).iterdir():
        loader = _get_loader(fp)
        docs.extend(loader.load())

    # 2. 文本切分：按字符分割
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    # 3. 本地 embeddings
    # 本地模型已事先下载到 ./models/all-MiniLM-L6-v2
    embeddings = HuggingFaceEmbeddings(
        model_name="./models/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="./models/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 4. 构建并持久化向量库
    vectordb = Chroma.from_documents(
        splits,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    return vectordb
