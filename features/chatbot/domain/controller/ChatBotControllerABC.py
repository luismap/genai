


from abc import ABC, abstractmethod

from features.chatbot.data.models.ChatBotModel import ChatBotModel


class ChatBotControllerABC(ABC):
    @abstractmethod
    def ask(question:str) -> ChatBotModel:
        """given a question, return a chatbot model
        with an answer to the question

        Args:
            question (str): the question

        Returns:
            ChatBotModel: chatbot model 
        """
        pass