import jax.numpy as np
import jax
from typing import Mapping


def lm_loss_fn(forward_fn,
               vocab_size: int,
               params,
               rng,
               data: Mapping[str, np.ndarray],
               is_training: bool = True) -> np.ndarray:
    """Compute the loss on data wrt params."""
    logits = forward_fn(params, rng, data, is_training)
    targets = jax.nn.one_hot(data['target'], vocab_size)
    assert logits.shape == targets.shape

    mask = np.greater(data['obs'], 0)
    loss = -np.sum(targets * jax.nn.log_softmax(logits), axis=-1)
    loss = np.sum(loss * mask) / np.sum(mask)

    return loss