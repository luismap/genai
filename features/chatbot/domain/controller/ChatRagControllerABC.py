from abc import ABC, abstractmethod
from typing import List

from features.chatbot.data.models.ChatBotModel import ChatBotReadModel


class ChatBotControllerABC(ABC):
    @abstractmethod
    def chat_rag(question:str) -> ChatBotReadModel:
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