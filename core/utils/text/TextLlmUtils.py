from langchain.document_loaders.text import TextLoader
from langchain.schema.document import Document
from langchain.text_splitter import CharacterTextSplitter
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import WebBaseLoader

class TextLlmUtils:
    """Utils class for interacting with different methods needed by the llm
    to handle text data
    """
    def loader(path: str) -> List[Document]:
        """given a file path, retrieve a list of (langchain) Documents
        Args:
            path (str): local path

        Returns:
            List[Document]: a list of documents
        """
        loader = TextLoader(path)
        docs = loader.load()
        return docs
    
    def webloader(links: List[str]) -> List[Document]:
        """given a list of web links, parse and create a documents
        list.
        Args:
            links (List[str]): list of webpages

        Returns:
            List[Document]: a list of documents
        """
        loader = WebBaseLoader(links)
        docs = loader.load()
        return docs
    
    def split(docs: List[Document], 
              chunk_size = 10,
              chunk_overlap = 0) -> List[Document]:
        """given a list of documents, create splits for those documents base
        om some criteria like chunk_size and chunk_overlap

        Args:
            docs (List[Document]): list of documents
            chunk_size (int, optional): chunk size. Defaults to 10.
            chunk_overlap (int, optional): overlapping chunks. Defaults to 0.

        Returns:
            List[Document]: a list of chunked documents
        """
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(docs)
        return splits
    
    def hugginface_embeddings(model_name:str = "sentence-transformers/all-mpnet-base-v2",
                              model_kwargs: dict = {'device': 'cpu'},
                              encode_kwargs: dict = {'normalize_embeddings': False}
                             ) -> HuggingFaceEmbeddings:
        """create a huggingfaceembedding model to be use as an embedding algorithm

        Args:
            model_name (str, optional): embeddings model to use. Defaults to "sentence-transformers/all-mpnet-base-v2".
            model_kwargs (_type_, optional): model kwargs. Defaults to {'device': 'cpu'}.
            encode_kwargs (_type_, optional): encode kwargs. Defaults to {'normalize_embeddings': False}.

        Returns:
            HuggingFaceEmbeddings: an embedding from langchain lib
        """
        hfe = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return hfe
    
    def from_doc_to_text(docs:List[Document]) -> List[str]:
        return [ doc.page_content for doc in docs]