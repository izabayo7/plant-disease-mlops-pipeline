# 🌱 Plant Disease Detection MLOps Pipeline

An end-to-end MLOps pipeline for detecting plant diseases from leaf images. This project demonstrates a complete machine learning lifecycle, from data processing and model training to deployment, monitoring, and load testing.

## 🚀 Features

*   **State-of-the-Art Model**: EfficientNetB0 with Transfer Learning (98%+ expected accuracy).
*   **FastAPI Backend**: High-performance API for real-time inference.
*   **Interactive Web UI**: Modern dashboard for easy image uploading and system monitoring.
*   **MLOps Best Practices**:
    *   Data Versioning & Augmentation
    *   Experiment Tracking (TensorBoard, CSV Logs)
    *   Model Checkpointing & Early Stopping
    *   Docker Containerization
    *   Load Testing (Locust)
    *   System Monitoring (CPU, RAM, Inference Latency)

## 🛠️ Tech Stack

*   **ML Framework**: TensorFlow/Keras (Metal GPU Optimized for macOS)
*   **Backend**: FastAPI, Uvicorn
*   **Frontend**: HTML5, CSS3, JavaScript
*   **Containerization**: Docker, Docker Compose
*   **Testing**: Locust (Load Testing)
*   **Data**: PlantVillage Dataset (38 Classes)

## 🌐 Live Demo

*   **Web Dashboard**: [https://mlops.bwenge.rw/web/index.html](https://mlops.bwenge.rw/web/index.html)
*   **API Documentation**: [https://mlops.bwenge.rw/docs](https://mlops.bwenge.rw/docs)

## 📂 Project Structure

```
├── api/                 # FastAPI backend
│   └── main.py
├── web/                 # Frontend UI
│   ├── index.html
│   ├── style.css
│   └── app.js
├── data/                # Dataset directory
│   ├── train/           # Training data
│   └── test/            # Test/Validation data
├── models/              # Saved models and logs
│   ├── final/           # Final .keras model
│   └── checkpoints/     # Training checkpoints
├── notebook/            # Training notebook
│   └── plant_disease_training.ipynb
├── load_testing/        # Locust load tests
│   └── locustfile.py
├── scripts/             # Utility scripts
├── visualizations/      # Generated plots (Confusion Matrix, etc.)
├── Dockerfile           # Docker image config
├── docker-compose.yml   # Docker services config
└── requirements.txt     # Python dependencies
```

## ⚡️ Quick Start

### 1. Local Setup

**Prerequisites**: Python 3.9+

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd plant-disease-mlops-pipeline
    ```

2.  **Create Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train the Model**:
    *   Open `notebook/plant_disease_training.ipynb`
    *   Run all cells to train and save the model to `models/final/plant_disease_model.keras`

5.  **Run the App**:
    ```bash
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   **Web UI**: Open [http://localhost:8000/web/index.html](http://localhost:8000/web/index.html)
    *   **API Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Docker Deployment

1.  **Build and Run**:
    ```bash
    docker-compose up --build
    ```
2.  Access the application at [http://localhost:8000/web/index.html](http://localhost:8000/web/index.html).

## 🧪 Load Testing

We use **Locust** to simulate user traffic and test system stability.

1.  **Start the API** (Local or Docker).
2.  **Run Locust**:
    ```bash
    locust -f load_testing/locustfile.py
    ```
3.  Open [http://localhost:8089](http://localhost:8089) to configure and start the test.

## 📊 Evaluation Metrics

The model is evaluated on:
*   **Accuracy**: Overall correctness.
*   **Precision/Recall/F1-Score**: Weighted and per-class metrics.
*   **Confusion Matrix**: To identify misclassifications.
*   **GradCAM**: Visual attention maps to interpret model focus.

*Metrics and visualizations are saved in `models/final/` and `visualizations/` after training.*

## 📝 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/predict` | Predict disease from a single image file. |
| `POST` | `/predict-batch` | Predict diseases from multiple images. |
| `GET` | `/monitoring` | Get system uptime, request counts, and hardware usage. |
| `GET` | `/health` | Health check probe. |
| `POST` | `/upload-data` | Upload new training data (images or ZIPs). |
| `POST` | `/retrain` | Trigger model retraining on new + original data. |

## 👤 Author

**Cedric Izabayo**
*   Project for ALU MLOps Assignment
