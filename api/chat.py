from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from models.schemas import ChatRequest, ChatResponse
from models.init_chat import Chat

from core.logging_logger import setup_logger
logger = setup_logger(__name__)

router = APIRouter()
chat_handler = Chat()

@router.post("", response_model=ChatResponse)
async def process_chat(chat_request: ChatRequest):
    try:
        prompt = chat_request.prompt
        file_id_input = chat_request.file_id

        response, file_url, references, follow_up_question = await chat_handler.generate_response(
            chat_request.chat_options, prompt, file_id_input
        )

        return JSONResponse(content={
            "response": response,
            "file_url": file_url,
            "created_at": chat_handler.now.isoformat(),
            "references": references,
            "follow_up_question": follow_up_question,
        }, status_code=200)

    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
