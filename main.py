from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np
import os
from tempfile import NamedTemporaryFile

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load your Keras model
model = load_model('model_vgg19.h5')

def predict(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        classes = model.predict(img_data)
        malignant = float(classes[0, 0])
        normal = float(classes[0, 1])

        return malignant, normal
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

def suggest_treatment(prediction, malignant_prob):
    if prediction == 'normal':
        return "No cancer detected. Regular check-ups and maintaining a healthy lifestyle are recommended."
    else:
        if malignant_prob < 0.7:
            return "Early-stage cancer detected. Consult with an oncologist for a detailed treatment plan. Options may include surgery, radiation therapy, or chemotherapy."
        elif 0.7 <= malignant_prob < 0.9:
            return "Advanced cancer detected. Immediate consultation with an oncologist is crucial. Treatment may involve a combination of surgery, chemotherapy, radiation therapy, and possibly immunotherapy."
        else:
            return "Highly aggressive cancer detected. Urgent consultation with a specialized oncology team is required. Treatment will likely involve aggressive combination therapy and possibly clinical trials."

@app.post("/predict")
async def predict_image(file: UploadFile):
    try:
        with NamedTemporaryFile(delete=False) as temp_image:
            contents = await file.read()
            temp_image.write(contents)
            temp_image_path = temp_image.name

        malignant, normal = predict(temp_image_path)

        if malignant > normal:
            prediction = 'malignant'
        else:
            prediction = 'normal'
        treatment_suggestion = suggest_treatment(prediction, malignant)

        return {
            "prediction": prediction,
            "malignant_prob": malignant,
            "normal_prob": normal,
            "treatment_suggestion": treatment_suggestion
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")
    finally:
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            os.remove(temp_image_path)