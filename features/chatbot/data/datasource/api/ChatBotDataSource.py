
from abc import ABC, abstractmethod
from typing import List, Tuple

from features.chatbot.data.models.ChatBotModel import ChatBotPayloadModel, ChatBotReadModel


class ChatBotDataSource(ABC):
    @abstractmethod
    def is_available() -> bool:
        pass

    @abstractmethod
    def generate_base_answer(question: str) -> ChatBotReadModel:
        pass

    @abstractmethod
    def chat(question:List[ChatBotPayloadModel]) -> List[ChatBotReadModel]:
        pass

    @abstractmethod
    def clean_user_history(user_id: str) -> bool:
        pass

    @abstractmethod
    def _add_user(user_name: str):
        pass