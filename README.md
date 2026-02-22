# MLOps Pipeline — Cats vs Dogs Classifier

This repository contains an end-to-end MLOps pipeline for binary image classification (Cats vs Dogs). I built this project as part of my M.Tech in AI & ML at BITS Pilani (Course: S1-25_AIMLCZG523).

The objective was to go beyond model training and implement a complete production-style workflow including data versioning, experiment tracking, containerization, CI/CD automation, deployment, and monitoring.

---

## Quick Links

Docker Hub Image:  
shaananshu2024ab05201/cats-dogs-classifier:latest  

CI/CD Pipeline Runs:  
See the “Actions” tab in this repository  

---

## Project Structure

```
mlops-cats-dogs/
├── src/
│   ├── preprocess.py        # Image resizing, validation, and splitting
│   ├── train.py             # Model training with MLflow tracking
│   └── app.py               # FastAPI inference service
├── tests/
│   ├── test_app.py
│   └── test_preprocess.py
├── models/
│   └── model.pt             # Trained weights
├── data/
│   └── processed/           # Train/val/test splits (tracked using DVC)
├── .github/workflows/
│   └── ci-cd.yml
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
└── requirements.txt
```

---

## Pipeline Overview

On every push to the main branch, the following steps run automatically:

1. Run unit tests using pytest  
2. Build Docker image  
3. Push image to Docker Hub  
4. Deploy using Docker Compose  
5. Execute smoke test (/health endpoint)  

If any step fails, the deployment is stopped.

---

## M1: Model Development & Experiment Tracking

Model:
- MobileNetV2 pretrained on ImageNet
- Fine-tuned for binary classification (cats vs dogs)
- Trained on CPU

Results:
- Test Accuracy: 86%
- Confusion Matrix: [[46, 4], [10, 40]]

Run locally:

```bash
python src/preprocess.py
python src/train.py
```

View experiments in MLflow:

```bash
mlflow ui
# Open http://localhost:5000
```

Tracked items:
- Parameters (epochs, batch size, learning rate, model)
- Metrics per epoch (train loss, validation accuracy)
- Final test accuracy
- Model artifact (model.pt)

DVC usage:

```bash
dvc push
dvc pull
```

---

## M2: Model Packaging & Containerization

Build and run locally:

```bash
docker build -t shaananshu2024ab05201/cats-dogs-classifier:latest .
docker run -d -p 8000:8000 --name cats-dogs-api shaananshu2024ab05201/cats-dogs-classifier:latest
```

API Endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health  | GET    | Returns service and model status |
| /predict | POST   | Accepts image file and returns label + probabilities |
| /metrics | GET    | Exposes Prometheus metrics |

Test API manually:

Health check:

```bash
curl http://localhost:8000/health
```

Prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@your_image.jpg"
```

---

## M3: CI Pipeline — GitHub Actions

The CI pipeline triggers automatically on every push to main.

Jobs:

1. test-and-build  
   - Install dependencies  
   - Run pytest  
   - Build Docker image  
   - Push image to Docker Hub  

2. deploy  
   - Pull latest image  
   - Deploy using Docker Compose  
   - Run smoke test  

Run tests locally:

```bash
pytest tests/ -v
```

---

## M4: CD Pipeline & Deployment

Deploy locally:

```bash
docker compose up -d
docker ps
```

Manual smoke test:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status": "ok", "model_loaded": true}
```

If the health check fails during CI/CD, the pipeline stops and the deployment is rejected.

---

## M5: Monitoring & Logging

View logs:

```bash
docker logs cats-dogs-api
```

Logs include:
- Model load events  
- Health checks  
- Prediction requests  
- Inference latency  

Prometheus metrics endpoint:

```
http://localhost:8000/metrics
```

Tracked metrics:
- request_count_total  
- request_latency_seconds (histogram)  

---

## Tech Stack

| Component           | Tool/Version                |
|---------------------|-----------------------------|
| Model               | MobileNetV2 (PyTorch 2.2.2) |
| Experiment Tracking | MLflow 2.12.2               |
| Data Versioning     | DVC 3.50.1                  |
| API Framework       | FastAPI 0.111.0             |
| Containerization    | Docker + Docker Compose     |
| CI/CD               | GitHub Actions              |
| Monitoring          | Prometheus client 0.20.0    |
| Testing             | pytest 8.1.1                |
| Registry            | Docker Hub                  |

---

All source code, configuration files, DVC tracking files, and trained model artifacts are included in this repository.