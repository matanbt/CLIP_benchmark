import torch

from src.models.mlp import MLPAligner
from src.models.transformer import TransformerEncoderAligner


def initialize_aligner_model(model_class_name: str, **kwargs) -> torch.nn:
    if model_class_name in ['MLP', 'MLPAligner']:
        return MLPAligner(**kwargs)
    elif model_class_name == 'TransformerEncoderAligner':
        return TransformerEncoderAligner(**kwargs)

    raise ValueError(f"Unknown model class name: {model_class_name}")
