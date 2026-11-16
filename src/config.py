import os
import yaml
from pathlib import Path


def load_config(path: str = None):
    cfg_path = Path(path) if path else Path('config/config.yaml')
    with cfg_path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # override with env
    raw = os.getenv('RAW_DATA_DIR')
    if raw:
        cfg['paths']['raw_data_dir'] = raw
    proc = os.getenv('PROCESSED_DIR')
    if proc:
        cfg['paths']['processed_dir'] = proc
    text_model = os.getenv('TEXT_EMBEDDING_MODEL') or os.getenv('MODEL_NAME')
    if text_model:
        cfg['model']['text_embedding_model'] = text_model
    image_model = os.getenv('IMAGE_MODEL')
    if image_model:
        cfg['model']['image_model'] = image_model
    device = os.getenv('DEVICE')
    if device:
        cfg['model']['device'] = device

    return cfg
