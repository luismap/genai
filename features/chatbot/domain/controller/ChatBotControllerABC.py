


from abc import ABC, abstractmethod

from features.chatbot.data.models.ChatBotModel import ChatBotReadModel


class ChatBotControllerABC(ABC):
    @abstractmethod
    def chat(question:str) -> ChatBotReadModel:
        """given a question, return a chatbot model
        with an answer to the question

        Args:
            question (str): the question

        Returns:
            ChatBotModel: chatbot model 
        """
        pass