from typing import List
from pydantic import BaseModel

class ChatBot(BaseModel):
    question: str
    chat_history: List[(str,str)]
    model_use: str