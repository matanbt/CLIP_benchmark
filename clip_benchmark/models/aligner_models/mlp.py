from torch import nn
import torch


class MLPAligner(nn.Module):
    def __init__(self, source_emb_dim: int, target_emb_dim: int,
                 n_hidden_layers: int = 0,
                 hidden_dim: int = None, **kwargs):
        """

        :param source_emb_dim: dimension of the source embedding to project
        :param target_emb_dim: dimension of the target embedding
        :param n_hidden_layers: model's hidden layer count. `0` means linear projection.
        :param hidden_dim: hidden layer's dimension. Must be specified if `n_hidden_layers > 0`.
        """
        super().__init__()
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
