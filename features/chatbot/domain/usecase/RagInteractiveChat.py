from typing import List
from features.chatbot.data.controller.ChatRagController import ChatRagController
from features.chatbot.data.models.ChatRagModel import (
    ChatRagPayloadModel,
    ChatRagReadModel,
)
from langchain.schema import Document


class RagInteractiveChat:
    def __init__(self, chatbot_controller: ChatRagController) -> None:
        self._chatbot_ctr = chatbot_controller

    def ask(self, crpms=List[ChatRagPayloadModel]) -> List[ChatRagReadModel]:
        answer = self._chatbot_ctr.chat_rag(crpms=crpms)
        return answer

    def load_text_from_local(self, path: str, user_id: str) -> bool:
        loaded = self._chatbot_ctr.load_text_from_local(path, user_id)
        return loaded

    def load_from_web(self, links: List[str], user_id: str) -> bool:
        web_loaded = self._chatbot_ctr.load_from_web(links, user_id)
        return web_loaded

    def clean_context(self, user_id=str) -> bool:
        return self._chatbot_ctr.clean_context(user_id=user_id)

    def get_context_length(self, user_id=str) -> bool:
        return self._chatbot_ctr.get_context_length(user_id=user_id)

    def similarity_search(self, content=str, user_id=str) -> List[Document]:
        return self._chatbot_ctr.similarity_search(content=content, user_id=user_id)
