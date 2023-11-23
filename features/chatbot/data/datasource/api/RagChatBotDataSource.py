
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
                 get_history:bool,
                 user_id: str) -> ChatRagReadModel:
        pass

    @abstractmethod
    def clean_user_history(user_id: str) -> bool:
        pass

    @abstractmethod
    def _add_user(user_name: str):
        pass