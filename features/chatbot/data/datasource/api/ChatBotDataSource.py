
from abc import ABC, abstractmethod

from features.chatbot.data.models.ChatBotModel import ChatBotReadModel


class ChatBotDataSource(ABC):
    @abstractmethod
    def generate_base_answer(question: str) -> ChatBotReadModel:
        pass

    @abstractmethod
    def chat(question: str, chat_bot_model: ChatBotReadModel) -> ChatBotReadModel:
        pass