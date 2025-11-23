from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
from datetime import datetime
from typing import List
import psutil
import time
import shutil
from .retrain import retrain_model

# Constants
NEW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'new_data')
os.makedirs(NEW_DATA_DIR, exist_ok=True)
TRAIN_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'train')


# Model paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'final', 'plant_disease_model.keras')
RETRAINED_MODEL_PATH = MODEL_PATH.replace('.keras', '_retrained_latest.keras')
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'class_names.json')

# Global variables for model and class names (will be reloaded after retraining)
model = None
class_names = []

def load_model_and_classes():
    """Load or reload the model and class names"""
    global model, class_names
    
    print("Loading model...")
    try:
        # Prefer retrained model if it exists, otherwise use original
        if os.path.exists(RETRAINED_MODEL_PATH):
            model = tf.keras.models.load_model(RETRAINED_MODEL_PATH)
            print(f"✅ Loaded RETRAINED model from {RETRAINED_MODEL_PATH}")
        elif os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"✅ Loaded ORIGINAL model from {MODEL_PATH}")
        else:
            print(f"❌ Model not found at {MODEL_PATH}. Please run training notebook first.")
            model = None
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    
    try:
        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, 'r') as f:
                class_names = json.load(f)
        else:
            print(f"Class names not found at {CLASS_NAMES_PATH}. Using dummy classes.")
            class_names = [f"class_{i}" for i in range(38)]
    except Exception as e:
        print(f"Error loading class names: {e}")
        class_names = [f"class_{i}" for i in range(38)]
    
    return model, class_names

# Load model and classes on startup
load_model_and_classes()

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Detection API",
    description="MLOps Pipeline for Plant Disease Classification",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/web", StaticFiles(directory="web"), name="web")

# Global metrics
START_TIME = datetime.now()
REQUEST_COUNT = 0
PREDICTION_TIMES = []

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Plant Disease Detection API",
        "status": "online",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict-batch",
            "health": "/health",
            "monitoring": "/monitoring"
        }
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict plant disease from single image"""
    global REQUEST_COUNT, PREDICTION_TIMES
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess
        img_array = preprocess_image(image)
        
        # Predict with timing
        start_time = time.time()
        predictions = model.predict(img_array, verbose=0)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Get top prediction
        pred_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][pred_idx])
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3 = [
            {
                "disease": class_names[idx],
                "confidence": float(predictions[0][idx])
            }
            for idx in top_3_idx
        ]
        
        # Update metrics
        REQUEST_COUNT += 1
        PREDICTION_TIMES.append(inference_time)
        
        return {
            "success": True,
            "prediction": {
                "disease": class_names[pred_idx],
                "confidence": confidence,
                "top_3": top_3
            },
            "inference_time_ms": round(inference_time, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """Predict plant diseases from multiple images"""
    global REQUEST_COUNT, PREDICTION_TIMES
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        results = []
        total_start = time.time()
        
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            img_array = preprocess_image(image)
            
            start_time = time.time()
            predictions = model.predict(img_array, verbose=0)
            inference_time = (time.time() - start_time) * 1000
            
            pred_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][pred_idx])
            
            results.append({
                "filename": file.filename,
                "disease": class_names[pred_idx],
                "confidence": confidence,
                "inference_time_ms": round(inference_time, 2)
            })
            
            PREDICTION_TIMES.append(inference_time)
        
        total_time = (time.time() - total_start) * 1000
        REQUEST_COUNT += len(files)
        
        return {
            "success": True,
            "count": len(results),
            "predictions": results,
            "total_time_ms": round(total_time, 2),
            "average_time_ms": round(total_time / len(results), 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/monitoring")
async def monitoring():
    """Get system and model monitoring metrics"""
    global REQUEST_COUNT, PREDICTION_TIMES, START_TIME
    
    uptime = (datetime.now() - START_TIME).total_seconds()
    
    # Calculate statistics
    avg_inference_time = np.mean(PREDICTION_TIMES) if PREDICTION_TIMES else 0
    p50_inference_time = np.percentile(PREDICTION_TIMES, 50) if PREDICTION_TIMES else 0
    p95_inference_time = np.percentile(PREDICTION_TIMES, 95) if PREDICTION_TIMES else 0
    p99_inference_time = np.percentile(PREDICTION_TIMES, 99) if PREDICTION_TIMES else 0
    
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    return {
        "uptime_seconds": round(uptime, 2),
        "uptime_human": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
        "total_requests": REQUEST_COUNT,
        "requests_per_second": round(REQUEST_COUNT / uptime, 2) if uptime > 0 else 0,
        "inference_metrics": {
            "count": len(PREDICTION_TIMES),
            "average_ms": round(avg_inference_time, 2),
            "p50_ms": round(p50_inference_time, 2),
            "p95_ms": round(p95_inference_time, 2),
            "p99_ms": round(p99_inference_time, 2),
            "min_ms": round(min(PREDICTION_TIMES), 2) if PREDICTION_TIMES else 0,
            "max_ms": round(max(PREDICTION_TIMES), 2) if PREDICTION_TIMES else 0
        },
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2)
        },
        "model_info": {
            "num_classes": len(class_names),
            "version": "1.0.0"
        }
    }



@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...), label: str = Form(...)):
    """Upload new data for retraining - supports single images or zip files"""
    try:
        # Create label directory if not exists
        label_dir = os.path.join(NEW_DATA_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        
        # Check if it's a zip file
        if file.filename.endswith('.zip'):
            # Save zip temporarily
            import zipfile
            zip_path = os.path.join(NEW_DATA_DIR, file.filename)
            with open(zip_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Extract images from zip (flatten structure - no nested folders)
            extracted_count = 0
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.namelist():
                    # Skip directories and __MACOSX files
                    if file_info.endswith('/') or '__MACOSX' in file_info or file_info.startswith('.'):
                        continue
                        
                    if file_info.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Read the file from zip
                        file_data = zip_ref.read(file_info)
                        
                        # Extract just the filename (no path)
                        filename = os.path.basename(file_info)
                        
                        # Write directly to label directory
                        output_path = os.path.join(label_dir, filename)
                        with open(output_path, 'wb') as img_file:
                            img_file.write(file_data)
                        
                        extracted_count += 1
            
            # Clean up zip file
            os.remove(zip_path)
            return {"message": f"Extracted {extracted_count} images to {label}"}
        else:
            # Single file upload
            file_path = os.path.join(label_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            return {"message": f"File {file.filename} uploaded successfully to {label}"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    def retrain_and_reload():
        """Retrain the model and reload it"""
        success, result = retrain_model(MODEL_PATH, NEW_DATA_DIR)
        if success:
            print("🔄 Reloading model after successful retraining...")
            load_model_and_classes()
            print("✅ Model reloaded successfully!")
        else:
            print(f"❌ Retraining failed: {result}")
    
    background_tasks.add_task(retrain_and_reload)
    return {"message": "Retraining process started in background"}

@app.get("/stats")
async def get_stats():
    """Get dataset and model statistics"""
    stats = {
        "training_data": {},
        "new_data": {},
        "total_images": 0
    }
    
    # Count training data
    if os.path.exists(TRAIN_DATA_DIR):
        for class_name in os.listdir(TRAIN_DATA_DIR):
            class_dir = os.path.join(TRAIN_DATA_DIR, class_name)
            if os.path.isdir(class_dir):
                count = len(os.listdir(class_dir))
                stats["training_data"][class_name] = count
                stats["total_images"] += count
                
    # Count new data
    if os.path.exists(NEW_DATA_DIR):
        for class_name in os.listdir(NEW_DATA_DIR):
            class_dir = os.path.join(NEW_DATA_DIR, class_name)
            if os.path.isdir(class_dir):
                count = len(os.listdir(class_dir))
                stats["new_data"][class_name] = count
                
    return stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
