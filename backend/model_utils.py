import tensorflow as tf
import numpy as np
from PIL import Image
import io

CLASS_NAMES = [
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

model = None

def load_model(model_path: str):
    """Load the trained model from disk."""
    global model
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✓ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise


def preprocess_image(image_bytes: bytes, target_size=(224, 224)):
    """
    Preprocess image for model prediction.
    
    Args:
        image_bytes: Raw image bytes from uploaded file
        target_size: Target size for the image (width, height)
    
    Returns:
        Preprocessed numpy array ready for model prediction
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize(target_size)
        
        img_array = np.array(img)
        
        img_array = np.expand_dims(img_array, axis=0)
        
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        return img_array
    
    except Exception as e:
        print(f"✗ Error preprocessing image: {e}")
        raise


def predict_disease(image_bytes: bytes):
    """
    Predict disease from leaf image.
    
    Args:
        image_bytes: Raw image bytes from uploaded file
    
    Returns:
        Dictionary with prediction results including class name and confidence
    """
    global model
    
    if model is None:
        raise ValueError("Model not loaded. Call load_model() first.")
    
    try:
        print(f"Preprocessing image ({len(image_bytes)} bytes)...")
        processed_image = preprocess_image(image_bytes)
        print(f"Image preprocessed: shape={processed_image.shape}")
        
        print("Making prediction...")
        predictions = model.predict(processed_image, verbose=0)
        print(f"Prediction complete: shape={predictions.shape}")
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        predicted_class = CLASS_NAMES[predicted_class_idx]
        print(f"Predicted: {predicted_class} ({confidence*100:.2f}%)")
        
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                "class": CLASS_NAMES[idx],
                "confidence": float(predictions[0][idx])
            }
            for idx in top_3_indices
        ]
        
        formatted_class = predicted_class.replace("_", " ").replace("  ", " - ")
        
        all_predictions = {
            CLASS_NAMES[i]: float(predictions[0][i] * 100)  
            for i in range(len(CLASS_NAMES))
        }
        
        return {
            "success": True,
            "predicted_class": predicted_class,
            "formatted_class": formatted_class,
            "confidence": confidence,
            "confidence_percentage": confidence * 100, 
            "top_3_predictions": top_3_predictions,
            "all_predictions": all_predictions
        }
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"✗ Error during prediction: {e}")
        print(f"Full traceback:\n{error_trace}")
        return {
            "success": False,
            "error": str(e)
        }


def get_model_info():
    """Get information about the loaded model."""
    global model
    
    if model is None:
        return {"loaded": False, "message": "No model loaded"}
    
    try:
        return {
            "loaded": True,
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "num_classes": len(CLASS_NAMES),
            "classes": CLASS_NAMES
        }
    except Exception as e:
        return {
            "loaded": True,
            "error": str(e)
        }
