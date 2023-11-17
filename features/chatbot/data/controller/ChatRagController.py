

import logging
from typing import List
from core.utils.MyUtils import MyUtils
from features.chatbot.data.datasource.RagLlama2DataSource import RagLlama2DataSource
from features.chatbot.data.datasource.api.RagChatBotDataSource import RagChatBotDataSource
from features.chatbot.data.datasource.api.VectorDbSource import VectorDbSource
from features.chatbot.data.models.ChatRagModel import ChatRagReadModel
from features.chatbot.domain.controller.ChatRagControllerABC import ChatRagControllerABC


class ChatRagController(ChatRagControllerABC):
    """
    Chatbot controller class. I will control interactions
    with the different datasources.

    Given a list of `ChatbotDataSource` it will check for the first
    available one, and will use it for inference
    """
    def __init__(self,
                 datasources: List[RagChatBotDataSource]
                 ) -> None:
        self._app_props = MyUtils.load_properties("general")["app"]
        self._app_state = self._app_props["env"]
        self._logger = logging.getLogger(self._app_props["logger"])
        self._chat_datasource = MyUtils.first(datasources, lambda ds: ds.is_available == True)
        self._vector_db: VectorDbSource = self._chat_datasource._vector_db

        self._logger.info("chatrag controller initialized")

        return None

    def chat_rag(self,question: str, history: bool = False) -> ChatRagReadModel:
        answer = self._chat_datasource.chat_rag(question, history)
        return answer

    def load_text_from_local(self,path: str) -> bool:
        loaded = self._vector_db.load_text_from_local(path)
        return loaded

    def load_from_web(self, links: List[str]) -> bool:
        web_loaded = self._vector_db.load_from_web(links)
        return web_loaded

    def clean_context(self) -> bool:
        if isinstance(self._chat_datasource, RagLlama2DataSource):
            self._chat_datasource._chat_rag_history = []
        return self._vector_db.clean_db()