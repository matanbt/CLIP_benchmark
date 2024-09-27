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
    # target_emb_model_name = 'openai/clip-vit-large-patch14'  # [HACK] TODO remove
    from transformers import CLIPModel, CLIPProcessor
    target_model = CLIPModel.from_pretrained(target_emb_model_name)
    target_model.to(device)
    target_processor = CLIPProcessor.from_pretrained(target_emb_model_name)
    target_tokenizer = AutoTokenizer.from_pretrained(target_emb_model_name, return_tensors="pt")

    # Load the source uni-modal model (e.g., E5 text-encoder)
    source_emb_model_name = metadata['dataset_metadata']['source_emb_model_name']
    source_model = SentenceTransformer(source_emb_model_name, device=device)
    source_tokenizer = source_model.tokenizer

    # Load aligner
    aligner_model = MLP(**metadata['model_kwargs'])  # TODO generalize the class (MLP) with `model_class_name`
    aligner_model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt")))
    aligner_model.eval()
    aligner_model.to(device)

    return (
        CombinedModel(source_model, target_model, aligner_model, model_type),
        partial(transform, target_processor=target_processor),
        partial(tokenize, tokenizer=target_tokenizer if model_type == 'target' else source_tokenizer)
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
        # target model will always embed the image
        # img = x['pixel_values']
        # Ensure the input tensor has the correct shape
        # if len(img.shape) == 5 and img.shape[1] == 1:
        #     img = img.squeeze(1)  # Remove the extra dimension
        # x['pixel_values'] = img

        return self.target_model.get_image_features(pixel_values=x)

    def forward(self, x):
        raise NotImplementedError

# [HACK] The following is a copy of the MLP module from the original code [TODO find a way to import]
class MLP(nn.Module):
    def __init__(self, source_emb_dim: int, target_emb_dim: int,
                 n_hidden_layers: int = 0,
                 hidden_dim: int = None, **kwargs):
        """

        :param source_emb_dim: dimension of the source embedding to project
        :param target_emb_dim: dimension of the target embedding
        :param n_hidden_layers: model's hidden layer count. `0` means linear projection.
        :param hidden_dim: hidden layer's dimension. Must be specified if `n_hidden_layers > 0`.
        """
        super(MLP, self).__init__()
        self.model_kwargs = dict(
            source_emb_dim=source_emb_dim,
            target_emb_dim=target_emb_dim,
            n_hidden_layers=n_hidden_layers,
            hidden_dim=hidden_dim,
            model_class_name=self.__class__.__name__
        )
        layers = []

        if n_hidden_layers == 0:
            layers.append(nn.Linear(source_emb_dim, target_emb_dim))
        else:
            if hidden_dim is None:
                raise ValueError("hidden_dim must be specified if hidden_layers > 0")

            layers.append(nn.Linear(source_emb_dim, hidden_dim))
            layers.append(nn.ReLU())

            for _ in range(n_hidden_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(hidden_dim, target_emb_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
