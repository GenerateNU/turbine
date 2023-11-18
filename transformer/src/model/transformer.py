from flax import linen as nn
import haiku as hk
from typing import Optional
import jax
import jax.numpy as np

from attention import SelfAttention, PositionalEncoding, SelfAttentionHk


class Transformer(nn.Module):
    hidden_dim: int
    num_heads: int
    num_layers: int
    dropout_rate: float = 0.1

    def setup(self) -> None:
        self.attention_blocks: list[SelfAttentionBlock] = [SelfAttentionBlock(self.hidden_dim, self.num_heads, self.dropout_rate)
                                 for _ in range(self.num_layers)]
        self.feedforward_blocks: list[FeedforwardBlock] = [FeedforwardBlock(self.hidden_dim) for _ in range(self.num_layers)]

    def __call__(self, inputs, mask=None, train=True):
        x = inputs

        for i in range(self.num_layers):
            x = self.attention_blocks[i](x, mask=mask, train=train)
            x = self.feedforward_blocks[i](x, train=train)

        return x

class HkTransformer(hk.Module):
    """A transformer stack."""

    def __init__(self,
                 num_heads: int,
                 num_layers: int,
                 dropout_rate: float,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout_rate = dropout_rate

    def __call__(self,
                 h: np.ndarray,
                 mask: Optional[np.ndarray],
                 is_training: bool) -> np.ndarray:
        """Connects the transformer.
        Args:
          h: Inputs, [B, T, H].
          mask: Padding mask, [B, T].
          is_training: Whether we're training or not.
        Returns:
          Array of shape [B, T, H].
        """

        init_scale: float = 2. / self._num_layers
        dropout_rate: float = self._dropout_rate if is_training else 0.
        if mask is not None:
            mask = mask[:, None, None, :]

        for i in range(self._num_layers):
            h_norm: np.ndarray = layer_norm(h, name=f'h{i}_ln_1')
            h_attn: np.ndarray = SelfAttentionHk(
                num_heads=self._num_heads,
                key_size=64,
                w_init_scale=init_scale,
                name=f'h{i}_attn')(h_norm, mask=mask)
            h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            h = h + h_attn
            h_norm = layer_norm(h, name=f'h{i}_ln_2')
            h_dense: np.ndarray = DenseBlock(init_scale, name=f'h{i}_mlp')(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
            h = h + h_dense
        h = layer_norm(h, name='ln_f')

        return h


class SelfAttentionBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    dropout_rate: float

    def setup(self):
        self.attention = SelfAttention(num_heads=self.num_heads, head_dim=self.hidden_dim // self.num_heads, dropout_rate=self.dropout_rate)
        self.ln1 = nn.LayerNorm()
        self.mlp = nn.Dense(self.hidden_dim * 4)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.ln2 = nn.LayerNorm()

    def __call__(self, x, mask=None, train=True):
        x = x + PositionalEncoding(max_len=x.shape[1], dropout_rate=self.dropout_rate)(x, train=train)
        attention_output = self.attention(self.ln1(x), mask=mask, train=train)
        x = x + self.dropout(attention_output)
        mlp_output = self.mlp(self.ln2(x))
        x = x + self.dropout(mlp_output)

        return x

class FeedforwardBlock(nn.Module):
    hidden_dim: int
    dropout_rate: float

    def setup(self):
        self.mlp = nn.Dense(self.hidden_dim * 4)
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, train=True):
        mlp_output = self.mlp(x)
        x = x + self.dropout(mlp_output)

        return x


class DenseBlock(hk.Module):
    """A 2-layer MLP"""

    def __init__(self,
                 init_scale: float,
                 widening_factor: int = 4,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor

    def __call__(self, x: np.ndarray) -> np.ndarray:
        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        x = hk.Linear(self._widening_factor * hiddens, w_init=initializer)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(hiddens, w_init=initializer)(x)
    

def layer_norm(x: np.ndarray, name: Optional[str] = None) -> np.ndarray:
    """Apply a unique LayerNorm to x with default settings."""
    return hk.LayerNorm(axis=-1,
                        create_scale=True,
                        create_offset=True,
                        name=name)(x)