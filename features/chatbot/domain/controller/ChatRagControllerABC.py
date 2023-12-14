from abc import ABC, abstractmethod
from typing import List

from features.chatbot.data.models.ChatRagModel import ChatRagReadModel



class ChatRagControllerABC(ABC):
    @abstractmethod
    def chat_rag(question:str, history: bool) -> ChatRagReadModel:
        pass

    @abstractmethod
    def load_text_from_local(path: str) -> bool:
        pass

    @abstractmethod
    def load_from_web(links: List[str]) -> bool:
        pass

    @abstractmethod
    def clean_context() -> bool:
        pass

    @abstractmethod
    def get_context_length(user_id: str) -> int:
        pass