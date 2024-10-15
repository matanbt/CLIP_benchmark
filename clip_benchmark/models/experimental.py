import os
import json
from functools import partial

import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from transformers import AutoTokenizer


def load_experimental_models(
        cache_dir: str,  # the model's dir, e.g., `out/clip-to-e5--mock/`
        model_name: str,  # the model type, in `target`, `source`, `source+aligner`
        device='cuda',
        **kwargs,
):
    model_dir, model_type = cache_dir, model_name
    assert model_type in ['target', 'source', 'source+aligner'], f"{model_type=} is invalid!"

    # Load
    with open(os.path.join(model_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    # Load the multi-modal model (e.g., CLIP)
    target_emb_model_name = metadata['dataset_metadata']['target_emb_model_name']
    assert target_emb_model_name == 'openai/clip-vit-large-patch14', "Only CLIP is supported for now"
    from transformers import CLIPModel, CLIPProcessor
    target_model = CLIPModel.from_pretrained(target_emb_model_name)
    target_model.to(device)
    target_processor = CLIPProcessor.from_pretrained(target_emb_model_name)
    target_tokenizer = AutoTokenizer.from_pretrained(target_emb_model_name, return_tensors="pt")

    # Load the source uni-modal model (e.g., E5 text-encoder)
    source_emb_model_name = metadata['dataset_metadata']['source_emb_model_name']

    if source_emb_model_name == 'random_embeddings':
        emb_dim = 768
        from transformers import BertTokenizer
        source_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        embedding_mat = torch.load(f'random_embeddings_{emb_dim}.pt').to(device)

        def source_model(inputs):
            embs = embedding_mat[inputs['input_ids']].mean(dim=1)  # average over tokens within each sequence (pooling)
            embs = torch.nn.functional.normalize(embs, dim=-1)
            return {'sentence_embedding': embs}  # imitates SeT's API

    else:  # model is loadable with SeT
        source_model = SentenceTransformer(source_emb_model_name, device=device)
        source_tokenizer = source_model.tokenizer

    # Load aligner
    from .aligner_models import initialize_aligner_model
    aligner_model = initialize_aligner_model(**metadata['model_kwargs'])
    aligner_model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt"),
                                             map_location=torch.device(device)))
    aligner_model.eval()
    aligner_model.to(device)

    # Select text tokenizer:
    if model_type.startswith('source') and 'glove' in source_emb_model_name:
        import transformers
        text_tokenizer = lambda x: transformers.tokenization_utils_base.BatchEncoding(source_model.tokenize(x))
    elif model_type.startswith('source'):
        text_tokenizer = partial(tokenize, tokenizer=source_tokenizer)
    else:  # model_type == 'target'
        text_tokenizer = partial(tokenize, tokenizer=target_tokenizer)

    return (
        CombinedModel(source_model, target_model, aligner_model, model_type),
        partial(transform, target_processor=target_processor),
        text_tokenizer
    )


# Define the named functions to replace the local objects
def transform(x, target_processor):
    return target_processor(images=x, return_tensors="pt")['pixel_values'].squeeze(0)


def tokenize(x, tokenizer):
    return tokenizer(x, padding=True, truncation=True, return_tensors="pt")


# The combined model, to give the benchmark what is expects
class CombinedModel(nn.Module):
    def __init__(self, source_model, target_model, aligner_model, model_type):
        super(CombinedModel, self).__init__()
        self.source_model = source_model
        self.target_model = target_model
        self.aligner_model = aligner_model
        self.model_type = model_type

    def encode_text(self, x):
        # gets token inputs
        if self.model_type == 'source':
            emb = self.source_model(x)["sentence_embedding"]
            emb = torch.nn.functional.normalize(emb, dim=-1)
            return emb

        elif self.model_type == 'target':
            emb = self.target_model.get_text_features(**x)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            return emb

        elif self.model_type == 'source+aligner':
            emb = self.source_model(x)["sentence_embedding"]
            emb = torch.nn.functional.normalize(emb, dim=-1)
            return self.aligner_model(emb)

    def encode_image(self, x):
        return self.target_model.get_image_features(pixel_values=x)

    def forward(self, x):
        raise NotImplementedError
