

from typing import List
from features.chatbot.data.controller.ChatRagController import ChatRagController
from features.chatbot.data.models.ChatRagModel import ChatRagPayloadModel, ChatRagReadModel


class RagInteractiveChat:
    def __init__(self,
                 chatbot_controller: ChatRagController) -> None:
        self._chatbot_ctr = chatbot_controller
    
    def ask(self, crpms = List[ChatRagPayloadModel]) -> List[ChatRagReadModel]:
        answer = self._chatbot_ctr.chat_rag(crpms=crpms)
        return answer

    def load_text_from_local(self,path: str) -> bool:
        loaded = self._chatbot_ctr.load_text_from_local(path)
        return loaded

    def load_from_web(self, links: List[str]) -> bool:
        web_loaded = self._chatbot_ctr.load_from_web(links)
        return web_loaded

    def clean_context(self, user_id = str) -> bool:
        return self._chatbot_ctr.clean_context(user_id=user_id)
