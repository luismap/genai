


from abc import ABC, abstractmethod
from typing import List

from features.chatbot.data.models.ChatBotModel import ChatBotPayloadModel, ChatBotReadModel


class ChatBotControllerABC(ABC):
    @abstractmethod
    def chat(cb_payloads:List[ChatBotPayloadModel]) -> List[ChatBotReadModel]:
        """given a question, return a chatbot model
        with an answer to the question

        Args:
            question (str): the question

        Returns:
            ChatBotReadModel: chatbot model 
        """
        pass

    @abstractmethod
    def clean_context(user_id: str) -> bool:
        pass

    @abstractmethod
    def get_context_length(user_id: str) -> int:
        pass