

import logging
from typing import List
from core.utils.MyUtils import MyUtils
from features.chatbot.data.datasource.api.ChatBotDataSource import ChatBotDataSource
from features.chatbot.data.models.ChatBotModel import ChatBotModel
from features.chatbot.domain.controller.ChatBotControllerABC import ChatBotControllerABC


class ChatBotController(ChatBotControllerABC):
    def __init__(self,
                 datasources: List[ChatBotDataSource]
                 ) -> None:
        self.app_props = MyUtils.load_properties("general")["app"]
        self.app_state = self.app_props["env"]
        self.logger = logging.getLogger(self.app_props["logger"])
        self.chat_datasource: ChatBotDataSource = MyUtils.first(datasources, lambda ds: ds.is_available == True)
        self.logger.info("chatbot controller initialized")

        return None

    def chat(self, question: str) -> ChatBotModel:
        answer = self.chat_datasource.chat(question)
        return answer