

import logging
from typing import List
from core.utils.MyUtils import MyUtils
from features.chatbot.data.datasource.api.ChatBotDataSource import ChatBotDataSource
from features.chatbot.data.models.ChatBotModel import ChatBotReadModel
from features.chatbot.domain.controller.ChatBotControllerABC import ChatBotControllerABC


class ChatBotController(ChatBotControllerABC):
    """
    Chatbot controller class. I will control interactions
    with the different datasources.

    Given a list of `ChatbotDataSource` it will check for the first
    available one, and will use it for inference
    """
    def __init__(self,
                 datasources: List[ChatBotDataSource]
                 ) -> None:
        self.app_props = MyUtils.load_properties("general")["app"]
        self.app_state = self.app_props["env"]
        self.logger = logging.getLogger(self.app_props["logger"])
        self.chat_datasource: ChatBotDataSource = MyUtils.first(datasources, lambda ds: ds.is_available == True)
        self.logger.info("chatbot controller initialized")

        return None

    def chat(self, question: str) -> ChatBotReadModel:
        answer = self.chat_datasource.chat(question)
        return answer