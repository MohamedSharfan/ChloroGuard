from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from model_utils import load_model, predict_disease, get_model_info
import os

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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "plant_disease_mobilenetv2_deploy.keras")

@app.on_event("startup")
async def startup_event():
    """Load the model when the API starts."""
    try:
        if os.path.exists(MODEL_PATH):
            load_model(MODEL_PATH)
            print(f"✓ Model loaded from: {MODEL_PATH}")
        else:
            print(f"⚠ Warning: Model file not found at {MODEL_PATH}")
            print("  Please train your model first or update MODEL_PATH")
    except Exception as e:
        print(f"✗ Error loading model: {e}")


@app.get("/")
async def root():
    """Root endpoint - API health check."""
    return {
        "message": "ChloroGuard API is running",
        "status": "healthy",
        "endpoints": {
            "predict": "/predict (POST - upload image)",
            "health": "/health (GET)",
            "model_info": "/model/info (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_info = get_model_info()
    return {
        "status": "healthy",
        "model_loaded": model_info.get("loaded", False),
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH)
    }


@app.get("/model/info")
async def model_information():
    """Get information about the loaded model."""
    return get_model_info()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict plant disease from uploaded leaf image.
    
    Args:
        file: Uploaded image file (JPG, PNG, etc.)
    
    Returns:
        JSON with prediction results including disease class and confidence
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (jpg, png, etc.)"
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
            "raw_class": result["predicted_class"],
            "confidence": result["confidence_percentage"],
            "confidence_score": result["confidence"],
            "top_predictions": result["top_3_predictions"],
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
