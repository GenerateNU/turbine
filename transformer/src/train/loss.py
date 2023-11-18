import jax.numpy as np
import jax
from typing import Mapping


def lm_loss_fn(forward_fn,
               vocab_size: int,
               params,
               rng,
               data: Mapping[str, np.ndarray],
               is_training: bool = True,
               mask_prob: float = 0.15) -> np.ndarray:
    """Compute the loss on data wrt params."""
    logits = forward_fn(params, rng, data, is_training)
    targets = data['obs']
    assert logits.shape == targets.shape

    mask = (jax.random.uniform(rng, targets.shape) < mask_prob) & (targets > 0)
    masked_targets = np.where(mask, targets, -1)  # Set non-masked positions to -1
    masked_logits = np.where(mask, logits, 0.0)  # Zero out logits for non-masked positions
    loss = jax.nn.softmax_cross_entropy_with_logits(masked_logits, jax.nn.one_hot(masked_targets, vocab_size))
    loss = np.sum(loss) / np.sum(mask)

    return loss