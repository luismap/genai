from typing import List, Tuple
from pydantic import BaseModel

class ChatRag(BaseModel):
    question: str
    chat_history: List[Tuple[str,str]] = []
    model_use: str
    answer: str