import time
import logging
import io
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title='Cats vs Dogs Classifier')

REQUEST_COUNT   = Counter('request_count', 'Total prediction requests')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Prediction latency')

MODEL_PATH = Path('models/model.pt')
device     = torch.device('cpu')

def load_model():
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, 1)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    m.eval()
    return m

model = None

@app.on_event('startup')
def startup():
    global model
    model = load_model()
    logger.info('Model loaded successfully.')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': model is not None}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    REQUEST_COUNT.inc()
    start = time.time()
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(tensor).squeeze()
            prob  = torch.sigmoid(logit).item()
        label = 'dog' if prob > 0.5 else 'cat'
        latency = time.time() - start
        REQUEST_LATENCY.observe(latency)
        logger.info(f'Prediction: {label} | prob: {prob:.4f} | latency: {latency:.3f}s')
        return {'label': label, 'dog_probability': round(prob, 4), 'cat_probability': round(1 - prob, 4)}
    except Exception as e:
        logger.error(f'Prediction error: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/metrics')
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
