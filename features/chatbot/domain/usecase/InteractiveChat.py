

from typing import List
from features.chatbot.data.controller.ChatBotController import ChatBotController
from features.chatbot.data.models.ChatBotModel import ChatBotReadModel


class InteractiveChat:
    def __init__(self,
                 chatbot_controller: ChatBotController) -> None:
        self._chatbot_ctr = chatbot_controller

    def ask_me_something(self,question: str, history: bool = False) -> ChatBotReadModel:
        answer = self._chatbot_ctr.chat(question, history)
        return answer

    def clean_context(self) -> bool:
        return self._chatbot_ctr.clean_context()
