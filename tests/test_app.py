import io
import torch
from PIL import Image
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

with patch('src.app.load_model') as mock_load:
    mock_model = MagicMock()
    mock_model.return_value = torch.tensor([[0.8]])
    mock_load.return_value = mock_model
    from src.app import app

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'

def test_predict_returns_label():
    buf = io.BytesIO()
    Image.new('RGB', (224, 224)).save(buf, format='JPEG')
    buf.seek(0)
    with patch('src.app.model') as mock_model:
        mock_model.return_value = torch.tensor([[0.8]])
        r = client.post('/predict', files={'file': ('test.jpg', buf, 'image/jpeg')})
    assert r.status_code == 200
    assert 'label' in r.json()
