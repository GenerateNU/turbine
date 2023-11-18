from typing import Mapping
import jax.numpy as np
import haiku as hk
from ..model import embeddings, HkTransformer


def build_forward_fn(vocab_size: int, d_model: int, num_heads: int,
                     num_layers: int, dropout_rate: float):
    """Create the model's forward pass."""

    def forward_fn(data: Mapping[str, np.ndarray],
                   is_training: bool = True) -> np.ndarray:
        """Forward pass."""
        input_embeddings, input_mask = embeddings(data, vocab_size, d_model)

        transformer = HkTransformer(
            num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_rate)
        output_embeddings = transformer(input_embeddings, input_mask, is_training)

        # Reverse the embeddings (untied)
        return hk.Linear(vocab_size)(output_embeddings)

    return forward_fn