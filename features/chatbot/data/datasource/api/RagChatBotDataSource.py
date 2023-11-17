
from abc import ABC, abstractmethod

from features.chatbot.data.models.ChatRagModel import ChatRagReadModel



class RagChatBotDataSource(ABC):
    @abstractmethod
    def is_available() -> bool:
        pass

    @abstractmethod
    def generate_base_answer(question: str) -> ChatRagReadModel:
        pass

    @abstractmethod
    def chat_rag(question: str,
                 get_history:bool = False) -> ChatRagReadModel:
        pass