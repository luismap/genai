
from abc import ABC, abstractmethod
from typing import List
from features.chatbot.data.models.ChatBotModel import ChatRagPayloadModel

from features.chatbot.data.models.ChatRagModel import ChatRagReadModel



class RagChatBotDataSource(ABC):
    @abstractmethod
    def is_available() -> bool:
        pass

    @abstractmethod
    def generate_base_answer(question: str) -> ChatRagReadModel:
        pass

    @abstractmethod
    def chat_rag(chatrag_models: List[ChatRagPayloadModel]) -> ChatRagReadModel:
        pass

    @abstractmethod
    def clean_user_history(user_id: str) -> bool:
        pass

    @abstractmethod
    def _add_user(user_name: str):
        pass