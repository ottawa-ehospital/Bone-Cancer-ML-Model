import os
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np
from tempfile import NamedTemporaryFile
import uvicorn
import logging
from typing import Dict, Union
import tensorflow as tf
import requests
from urllib.parse import urlparse, parse_qs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cancer Detection API",
    description="API for detecting cancer in medical images using VGG19 model",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_google_drive_file_id(url: str) -> str:
    """Extract file ID from Google Drive URL"""
    if 'drive.google.com' not in url:
        return url
        
    parsed = urlparse(url)
    if 'id=' in url:
        return parse_qs(parsed.query)['id'][0]
    return parsed.path.split('/')[-2]

def download_from_google_drive():
    """Download the model file from Google Drive"""
    drive_url = os.getenv('MODEL_URL')
    if not drive_url:
        raise RuntimeError("MODEL_URL environment variable is not set")
    
    try:
        logger.info("Downloading model from Google Drive...")
        file_id = get_google_drive_file_id(drive_url)
        download_url = f"https://drive.google.com/uc?id={file_id}"
        
        local_model_path = "model_vgg19.h5"
        
        session = requests.Session()
        response = session.get(download_url, stream=True)
        response.raise_for_status()
        
        with open(local_model_path, "wb") as model_file:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    model_file.write(chunk)
                    
        logger.info("Model downloaded successfully")
        return local_model_path
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise RuntimeError(f"Failed to download model: {str(e)}")

# Initialize model at startup
try:
    model_path = download_from_google_drive()
    model = load_model(model_path)
    model.make_predict_function()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    raise RuntimeError("Failed to initialize model")

def predict(image_path: str) -> tuple[float, float]:
    """
    Perform prediction on the input image.
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        tuple: Probability scores for malignant and normal classes
    """
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        
        with tf.device('/CPU:0'):  # Force CPU usage for more stable deployment
            classes = model.predict(img_data)
            
        malignant = float(classes[0, 0])
        normal = float(classes[0, 1])
        return malignant, normal
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint to verify API status"""
    return {
        "message": "Bone Cancer Detection API is running",
        "status": "success",
        "version": "1.0.0"
    }

@app.post("/predict")
async def predict_image(file: UploadFile) -> Dict[str, Union[str, float]]:
    """
    Process uploaded image and return cancer prediction results.
    
    Args:
        file (UploadFile): Uploaded image file
        
    Returns:
        dict: Prediction results including class and probabilities
    """
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only PNG and JPEG images are supported")
    
    temp_image_path = None
    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_image:
            contents = await file.read()
            temp_image.write(contents)
            temp_image_path = temp_image.name
            
        malignant, normal = predict(temp_image_path)
        prediction = 'malignant' if malignant > normal else 'normal'
        
        return {
            "prediction": prediction,
            "malignant_prob": round(malignant, 4),
            "normal_prob": round(normal, 4),
            "confidence": round(max(malignant, normal) * 100, 2)
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}")

# Heroku deployment configuration
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    # Configure server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info",
        reload=False
    )