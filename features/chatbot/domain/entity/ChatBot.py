from typing import List, Tuple
from pydantic import BaseModel

class ChatBot(BaseModel):
    user_id: str = "default"
    question: str
    chat_history: List[Tuple[str,str]] = []
    batch_history: str = ""
    model_use: str
    answer: str

class ChatBotPayload(BaseModel):
    user_id: str
    question: str
    history: str