from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from core.firebase import db, bucket
from firebase_admin import firestore

import uuid
import os

from core.logging_logger import setup_logger
logger = setup_logger(__name__)

router = APIRouter()

@router.post("")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        file_path = f"users/{file_id}{file_extension}"

        logger.info(f"Uploading file: {file.filename} with ID: {file_id}")

        blob = bucket.blob(file_path)
        contents = await file.read()
        blob.upload_from_string(contents, content_type=file.content_type)

        logger.info(f"File uploaded to GCS: {file_path}")

        db.collection('files').document(file_id).set({
            'filename': file.filename,
            'content_type': file.content_type,
            'storage_path': file_path,
            'url': blob.public_url,
            'created_at': firestore.SERVER_TIMESTAMP
        })

        logger.info(f"File metadata saved to Firestore: {file_id}")

        return JSONResponse({
            "success": True,
            "file_id": file_id,
            "url": blob.public_url
        }, status_code=201)

    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
