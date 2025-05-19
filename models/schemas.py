from typing import Optional, List, Literal
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    prompt: str 
    file_id: Optional[str] = None
    chat_options: Literal[
        "General Macroeconomics",
        "2 Wheels",
        "4 Wheels",
        "Retail General",
        "Retail Beauty",
        "Retail FnB",
        "Retail Drugstore"
    ] = Field(default="General Macroeconomics")

class ChatResponse(BaseModel):
    response: str
    file_url: Optional[str] = None