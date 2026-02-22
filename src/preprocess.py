import os
import random
from pathlib import Path
from PIL import Image

def is_valid_image(filepath):
    try:
        img = Image.open(filepath)
        img.verify()
        return True
    except Exception:
        return False

def preprocess_dataset(
    src_dir: str,
    dst_dir: str,
    img_size: tuple = (224, 224),
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    sample_size: int = 500,
    seed: int = 42,
):
    random.seed(seed)
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    for split in ['train', 'val', 'test']:
        for cls in ['cat', 'dog']:
            (dst_dir / split / cls).mkdir(parents=True, exist_ok=True)

    for cls, folder in [('cat', 'Cat'), ('dog', 'Dog')]:
        images = [
            f for f in (src_dir / folder).iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and is_valid_image(f)
        ]
        random.shuffle(images)
        images = images[:sample_size]

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:],
        }

        for split, files in splits.items():
            for i, fp in enumerate(files):
                try:
                    img = Image.open(fp).convert('RGB').resize(img_size)
                    img.save(dst_dir / split / cls / f'{cls}_{i}.jpg')
                except Exception:
                    continue

    print('Preprocessing complete.')

if __name__ == '__main__':
    preprocess_dataset(
        src_dir=r'C:\Users\sanshu\Documents\archive\PetImages',
        dst_dir='data/processed',
    )
