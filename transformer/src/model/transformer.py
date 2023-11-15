import jax
import jax.numpy as np
from flax import linen as nn

from attention import SelfAttention, PositionalEncoding


class Transformer(nn.Module):
    hidden_dim: int
    num_heads: int
    num_layers: int
    dropout_rate: float = 0.1

    def setup(self):
        self.attention_blocks = [SelfAttentionBlock(self.hidden_dim, self.num_heads, self.dropout_rate)
                                 for _ in range(self.num_layers)]
        self.feedforward_blocks = [FeedforwardBlock(self.hidden_dim) for _ in range(self.num_layers)]

    def __call__(self, inputs, mask=None, train=True):
        x = inputs

        for i in range(self.num_layers):
            x = self.attention_blocks[i](x, mask=mask, train=train)
            x = self.feedforward_blocks[i](x, train=train)

        return x

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
