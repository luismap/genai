

from typing import List
from features.chatbot.data.controller.ChatBotController import ChatBotController
from features.chatbot.data.models.ChatBotModel import ChatBotReadModel


class InteractiveChat:
    def __init__(self,
                 chatbot_controller: ChatBotController) -> None:
        self._chatbot_ctr = chatbot_controller

    def ask_me_something(self,question: str) -> ChatBotReadModel:
        answer = self._chatbot_ctr.chat(question)
        return answer
    
    def ask_with_rag(self, question: str) -> ChatBotReadModel:
        answer = self._chatbot_ctr.chat_rag(question)
        return answer

    def load_text_from_local(self,path: str) -> bool:
        loaded = self._chatbot_ctr.load_text_from_local(path)
        return loaded

    def load_from_web(self, links: List[str]) -> bool:
        web_loaded = self._chatbot_ctr.load_from_web(links)
        return web_loaded

    def clean_context(self) -> bool:
        return self._chatbot_ctr.clean_context()
