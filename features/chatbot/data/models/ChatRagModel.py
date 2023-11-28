
from typing import List, Tuple
from pydantic import BaseModel
from datetime import datetime

from features.chatbot.domain.entity.ChatRag import ChatRag, ChatRagPayload

class ChatRagModel(ChatRag):
    date_created: str = str(datetime.now())

class ChatRagReadModel(ChatRag):
    date_read: str = str(datetime.now())


class ChatRagResponseModel(BaseModel):
    answer: str
    model_use: str
    question: str
    chat_history: List[Tuple[str,str]] = []

class ChatRagPayloadModel(ChatRagPayload):
    pass