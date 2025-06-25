from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os

router = APIRouter()

@router.post("/verify-product")
async def verify_product(
    image: UploadFile = File(...),
    product_id: str = Form(...)
):
    contents = await image.read()
    user_image = Image.open(io.BytesIO(contents)).convert("RGB")

    reference_path = f"app/reference_images/{product_id}.jpg"
    
    if not os.path.exists(reference_path):
        return JSONResponse({"error": "Reference image not found"}, status_code=404)

    reference_image = Image.open(reference_path).convert("RGB")

    match = user_image.size == reference_image.size
    confidence = 0.9 if match else 0.2

    return JSONResponse({
        "match": match,
        "confidence": confidence,
        "product_id": product_id
    })
