


from abc import ABC, abstractmethod
from typing import List

from features.chatbot.data.models.ChatBotModel import ChatBotReadModel


class ChatBotControllerABC(ABC):
    @abstractmethod
    def chat(question:str) -> ChatBotReadModel:
        """given a question, return a chatbot model
        with an answer to the question

        Args:
            question (str): the question

        Returns:
            ChatBotReadModel: chatbot model 
        """
        pass

    @abstractmethod
    def clean_context() -> bool:
        pass