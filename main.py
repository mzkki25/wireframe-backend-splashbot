from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import chat, upload, init_question

import uvicorn

app = FastAPI(title="SPLASHBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/upload", tags=["File Upload"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(init_question.router, prefix="/init_questions", tags=["Initial Questions"])

if __name__ == "__main__":
    import os
    uvicorn.run(port=int(os.environ.get("PORT", 8000)), host='0.0.0.0')