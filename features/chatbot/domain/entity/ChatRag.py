from typing import List, Tuple
from pydantic import BaseModel

class ChatRag(BaseModel):
    user_id: str
    question: str
    chat_history: List[Tuple[str,str]] = []
    model_use: str
    answer: str

class ChatRagPayload(BaseModel):
    user_id: str
    question: str
    history: bool = False