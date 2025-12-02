# 🌿 ChloroGuard - Plant Disease Detection Web App

AI-powered plant disease detection system using MobileNetV2 for Pepper, Tomato, and Potato plants.

## 🚀 Quick Start (Local Deployment)

### 1. Start the Backend API

```powershell
cd E:\chloroguard\ChloroGuard\backend
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

- **API**: http://localhost:8000
- **Interactive API Docs**: http://localhost:8000/docs

### 2. Open the Frontend

Simply open `frontend_quick/upload.html` in your browser:

```powershell
Start-Process "E:\chloroguard\ChloroGuard\frontend_quick\upload.html"
```

Or double-click the `upload.html` file.

## 📁 Project Structure

```
ChloroGuard/
├── backend/
│   ├── app.py              # FastAPI server
│   ├── model_utils.py      # Model loading & prediction
│   └── model/
│       └── mobilenetv2.py  # Model architecture
├── models/
│   └── plant_disease_mobilenetv2_deploy.keras  # Trained model
├── frontend_quick/
│   └── upload.html         # Web interface
└── notebooks/
    └── mobilenetv2.ipynb   # Training notebook
```

## 🌱 Supported Plants & Diseases

### 🌶️ **Pepper (2 classes)**

- Bacterial spot
- Healthy

### 🍅 **Tomato (10 classes)**

- Bacterial spot
- Early blight
- Late blight
- Healthy
- Leaf Mold
- Septoria leaf spot
- Spider mites
- Target Spot
- Tomato mosaic virus
- Yellow Leaf Curl Virus

### 🥔 **Potato (3 classes)**

- Early blight
- Late blight
- Healthy

## 🔧 API Endpoints

### GET `/`

Health check and API info

### GET `/health`

Server and model status

### POST `/predict`

Upload image, get top prediction + top 3

**Request:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@leaf_image.jpg"
```

**Response:**

```json
{
  "prediction": "Tomato Late blight",
  "confidence": "99.87%",
  "top_predictions": [...]
}
```

### POST `/predict/detailed`

Upload image, get predictions for all 15 classes

## 🌐 Cloud Deployment Options

### Option 1: Heroku (Free Tier Available)

```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create app
heroku create chloroguard-app

# Deploy
git push heroku main
```

### Option 2: Render.com (Free)

1. Connect your GitHub repo
2. Select "Web Service"
3. Build command: `pip install -r backend/requirements.txt`
4. Start command: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`

### Option 3: Railway.app (Free)

1. Connect GitHub repo
2. Auto-detects Python
3. Add start command: `cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT`

### Option 4: Vercel (Frontend) + Hugging Face Spaces (Backend)

- **Frontend**: Deploy `frontend_quick/upload.html` to Vercel
- **Backend**: Deploy FastAPI to Hugging Face Spaces with GPU

## 📊 Model Performance

- **Architecture**: MobileNetV2
- **Parameters**: 2.28M (lightweight!)
- **Training Accuracy**: ~91%
- **Validation Accuracy**: ~81%
- **Image Size**: 224x224
- **Training Dataset**: PlantVillage (20,638 images)

## 🛠️ Development

### Train New Model

```bash
# Open Jupyter notebook
jupyter notebook notebooks/mobilenetv2.ipynb

# Run cells 1-6 to train
# Run deployment cell to save model
```

### Test API Locally

```bash
# Test with curl
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_leaf.jpg"

# Or visit http://localhost:8000/docs for interactive testing
```

## 📦 Requirements

- Python 3.10+
- TensorFlow 2.20+
- FastAPI 0.116+
- Uvicorn

Install all dependencies:

```bash
pip install -r backend/requirements.txt
```

## 🐛 Troubleshooting

### Model won't load

- Run the deployment cell in the notebook to create `.keras` model
- Make sure `models/plant_disease_mobilenetv2_deploy.keras` exists

### CORS errors in browser

- Backend should already have CORS enabled for all origins
- Check that API is running on port 8000

### Low confidence predictions

- Make sure you're testing with Pepper/Tomato/Potato leaves only
- Model works best with clear images of individual leaves
- Ensure good lighting and leaf fills most of the frame

## 📝 License

MIT License - Feel free to use for learning and non-commercial projects!

## 👨‍💻 Author

Mohamed Sharfan
