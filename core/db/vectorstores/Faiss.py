
from typing import List
from langchain.vectorstores import FAISS
from langchain.schema.document import Document

class FaissLangchain:

    def from_documents(docs: List[Document],
                       encoder) -> FAISS:
        """create a vector store from the passed
        documents.

        Args:
            docs (List[str]): a list of documents to add to the 
            store
            encoder (_type_): encoder for this documents

        Returns:
            FAISS: a Faiss vector store
        """
        db = FAISS.from_documents(docs, encoder)
        return db