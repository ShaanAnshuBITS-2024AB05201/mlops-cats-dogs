from PIL import Image
from src.preprocess import is_valid_image

def test_valid_image(tmp_path):
    img_path = tmp_path / 'test.jpg'
    Image.new('RGB', (100, 100)).save(str(img_path))
    assert is_valid_image(str(img_path)) is True

def test_invalid_image(tmp_path):
    bad_path = tmp_path / 'bad.jpg'
    bad_path.write_bytes(b'not an image')
    assert is_valid_image(str(bad_path)) is False
