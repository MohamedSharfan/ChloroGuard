from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from .model_utils import  predict_disease, get_model_info
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = FastAPI(
    title="ChloroGuard API",
    description="Plant Disease Detection API using MobileNetV2",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus"
]

FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend_quick")
app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "plant_disease_mobilenetv2_deploy.keras")

@app.on_event("startup")
async def startup_event():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH, compile = False)
            print(f"Model loaded from: {MODEL_PATH}")
        else:
            print(f"Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")





@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))



@app.get("/health")
async def health_check():
    model_info = get_model_info()
    return {
        "status": "healthy",
        "model_loaded": model_info.get("loaded", False),
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH)
    }


@app.get("/model/info")
async def model_information():
    return get_model_info()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        image_bytes = await file.read()
        
        img = Image.open(io.BytesIO(image_bytes))

        if img.mode!= 'RGB':
            img = img.convert('RGB')
        img = img.resize((224,224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        
        
        global model
        if model is None:
            raise ValueError("Model is not loaded properly")
        
        try:
            prediction = model.predict(img_array, verbose = 0)
            print("Prediction complete")

            predicted_class_indx = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_indx])

            predicted_class = class_names[predicted_class_indx]
            top3_indices = np.argsort(prediction[0])[-3:][::-1]
            top3_prediction = [
                {
                    "class":class_names[indx],
                    "confidence": float(prediction[0][indx] * 100)
                } for indx in top3_indices
            ]

            formatted_class = predicted_class.replace('_',' ').replace(' ',' - ')

        except Exception as e:
            raise ValueError("Error during prediction: {e}")

        # result = predict_disease(image_bytes)
        
        
        
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": formatted_class,
            "raw_class": predicted_class,
            "confidence": confidence*100,
            "confidence_score": confidence,
            "top_predictions": top3_prediction,
            "status": "success"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/predict/detailed")
async def predict_detailed(file: UploadFile = File(...)):
    """
    Predict plant disease with detailed probability for all classes.
    
    Args:
        file: Uploaded image file
    
    Returns:
        JSON with detailed prediction results for all classes
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        image_bytes = await file.read()
        result = predict_disease(image_bytes)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {result.get('error', 'Unknown error')}"
            )
        
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": result["formatted_class"],
            "confidence": result["confidence_percentage"],
            "all_predictions": result["all_predictions"],
            "status": "success"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


if __name__ == "__main__":
    print("Starting ChloroGuard API...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Server will be available at: http://localhost:8000")
    print(f"API docs available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True  
    )
