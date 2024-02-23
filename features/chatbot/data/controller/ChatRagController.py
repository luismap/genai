import logging
from typing import List
from core.utils.MyUtils import MyUtils
from features.chatbot.data.datasource.api.RagChatBotDataSource import (
    RagChatBotDataSource,
)
from features.chatbot.data.datasource.api.VectorDbSource import VectorDbSource
from features.chatbot.data.models.ChatRagModel import (
    ChatRagPayloadModel,
    ChatRagReadModel,
)
from features.chatbot.domain.controller.ChatRagControllerABC import ChatRagControllerABC


class ChatRagController(ChatRagControllerABC):
    """
    Chatbot controller class. I will control interactions
    with the different datasources.

    Given a list of `ChatbotDataSource` it will check for the first
    available one, and will use it for inference
    """

    def __init__(self, datasources: List[RagChatBotDataSource]) -> None:
        self._app_props = MyUtils.load_properties("general")["app"]
        self._app_state = self._app_props["env"]
        self._logger = logging.getLogger(self._app_props["logger"])
        self._chat_datasource = MyUtils.first(
            datasources, lambda ds: ds.is_available == True
        )
        # deprecated: self._vector_db: VectorDbSource = self._chat_datasource._vector_db

        self._logger.info("chatrag controller initialized")

        return None

    def chat_rag(self, crpms=List[ChatRagPayloadModel]) -> List[ChatRagReadModel]:
        answers = self._chat_datasource.chat_rag(crpms)
        return answers

    def load_text_from_local(self, path: str, user_id: str) -> bool:
        loaded = self._chat_datasource._user_info[user_id][
            "vector_db"
        ].load_text_from_local(path)
        return loaded

    def load_from_web(self, links: List[str], user_id: str) -> bool:
        web_loaded = self._chat_datasource._user_info[user_id][
            "vector_db"
        ].load_from_web(links)
        return web_loaded

    def clean_context(self, user_id) -> bool:
        cleaned = self._chat_datasource.clean_user_history(user_id)
        return cleaned

    def get_context_length(self, user_id: str) -> int:
        parse_tuples = lambda t: t[0] + t[1]
        if user_id in self._chat_datasource._users:
            user_history = self._chat_datasource._user_info[user_id]["history"]
            history = " ".join(
                [parse_tuples(question_pair) for question_pair in user_history]
            )
            context_length = len(history)
        else:
            context_length = 0
        return context_length
