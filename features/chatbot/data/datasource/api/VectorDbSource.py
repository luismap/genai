

from abc import ABC, abstractmethod
from langchain.document_loaders.text import TextLoader
from langchain.schema.document import Document
from langchain.text_splitter import CharacterTextSplitter
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import WebBaseLoader

class VectorDbSource(ABC):
    @abstractmethod
    def load_text_from_local(path: str) -> bool:
        pass

    @abstractmethod
    def load_from_web(links: List[str]) -> bool:
        pass

    @abstractmethod
    def is_available() -> bool:
        pass

    @abstractmethod
    def clean_db() -> bool:
        pass
    
    @abstractmethod
    def similarity_search(question: str) -> List[Document]:
        pass