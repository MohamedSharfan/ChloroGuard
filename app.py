import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

CLASS_NAMES = [
    "Pepper Bell - Bacterial Spot",
    "Pepper Bell - Healthy",
    "Potato - Early Blight",
    "Potato - Healthy",
    "Potato - Late Blight",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Healthy",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites",
    "Tomato - Target Spot",
    "Tomato - Mosaic Virus",
    "Tomato - Yellow Leaf Curl Virus"
]

model = tf.keras.models.load_model('plant_disease_mobilenetv2_deploy.keras')

def predict_disease(image):
    img = Image.fromarray(image.astype('uint8'), 'RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx]) * 100
    
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    results = {CLASS_NAMES[idx]: float(predictions[0][idx] * 100) for idx in top_3_indices}
    
    return results

demo = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title="🌿 ChloroGuard - Plant Disease Detection",
    description="Upload a leaf image to detect diseases in Pepper, Potato, or Tomato plants",
    examples=[],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
