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
        self._app_props = MyUtils.load_properties("general")["app"]
        self._app_state = self._app_props["env"]
        self._logger = logging.getLogger(self._app_props["logger"])
        self._chat_datasource = MyUtils.first(datasources, lambda ds: ds.is_available == True)

        self._logger.info("chatbot controller initialized")

        return None

    def chat(self, question: str, history: bool = False) -> ChatBotReadModel:
        answer = self._chat_datasource.chat(question, history)
        return answer

    def clean_context(self) -> bool:
        return self._chat_datasource.clean_memory()