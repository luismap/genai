import logging
from typing import List
from core.utils.MyUtils import MyUtils
from features.chatbot.data.datasource.api.ChatBotDataSource import ChatBotDataSource
from features.chatbot.data.models.ChatBotModel import ChatBotPayloadModel, ChatBotReadModel
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

    def chat(self, cb_payloads: List[ChatBotPayloadModel]) -> List[ChatBotReadModel]:
        answer = self._chat_datasource.chat(cb_payloads)
        return answer

    def clean_context(self, user_id) -> bool:
        return self._chat_datasource.clean_user_history(user_id)
    
    def get_context_length(self, user_id: str) -> int:
        if user_id in self._chat_datasource._users:
            context_length = self._chat_datasource._generate_history(self._user_info[user_id]["history"])
        else:
            context_length = 0
        return context_length