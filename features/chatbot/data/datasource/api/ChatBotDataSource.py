
from abc import ABC, abstractmethod
from pydantic import BaseModel

from features.chatbot.data.models.ChatBotModel import ChatBotReadModel


class ChatBotDataSource(ABC):
    @abstractmethod
    def generate_answer(question: str) -> ChatBotReadModel:
        pass