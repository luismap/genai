

from typing import List
from features.chatbot.data.controller.ChatBotController import ChatBotController
from features.chatbot.data.models.ChatBotModel import ChatBotPayloadModel, ChatBotReadModel


class InteractiveChat:
    def __init__(self,
                 chatbot_controller: ChatBotController) -> None:
        self._chatbot_ctr = chatbot_controller

    def ask_me_something(self,cbpms: List[ChatBotPayloadModel]) -> List[ChatBotReadModel]:
        answer = self._chatbot_ctr.chat(cbpms)
        return answer

    def clean_context(self, user_id: str) -> bool:
        return self._chatbot_ctr.clean_context(user_id)
