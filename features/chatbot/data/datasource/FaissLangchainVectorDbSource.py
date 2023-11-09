

from typing import List
from features.chatbot.data.datasource.api.VectorDbSource import VectorDbSource
from core.db.vectorstores.Faiss import FaissLangchain
from langchain.schema import Document
from langchain.vectorstores.faiss import FAISS
from core.utils.text.TextLlmUtils import TextLlmUtils

class FaissLangchainVectorDbSource(VectorDbSource):
    def __init__(self) -> None:
        initial_doc = [Document(page_content="initial doc")]
        self._hfe = TextLlmUtils.hugginface_embeddings()
        self._vector_db: FAISS = FaissLangchain.from_documents(initial_doc,self._hfe)

    def load_text_from_local(self,path: str) -> bool:
        local_docs = TextLlmUtils.loader(path)
        local_docs_splits = TextLlmUtils.split(local_docs)
        self._vector_db.add_documents(local_docs_splits)
        return True
    
    def load_from_web(self,links: List[str]) -> bool:
        web_docs = TextLlmUtils.webloader(links)
        web_docs_splits = TextLlmUtils.split(web_docs)
        self._vector_db.add_documents(web_docs_splits)
        return True
    
    def is_available() -> bool:
        return True
    
    def clean_db(self) -> bool:
        initial_doc = [Document(page_content="initial doc")]
        self._vector_db: FAISS = FaissLangchain.from_documents(initial_doc,self._hfe)
        return True
    
    def similarity_search(self, question: str) -> List[Document]:
        search = self._vector_db.similarity_search(question)
        return search