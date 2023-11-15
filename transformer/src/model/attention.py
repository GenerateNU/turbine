import jax
import jax.numpy as np
from flax import linen as nn


class SelfAttention(nn.Module):
    num_heads: int
    head_dim: int
    dropout_rate: float

    def setup(self):
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, mask=None, train=True):
        qkv = nn.Dense(self.num_heads * 3 * self.head_dim)(x)
        qkv = np.split(qkv, 3, axis=-1)
        q, k, v = [np.reshape(t, (x.shape[0], -1, self.num_heads, self.head_dim)) for t in qkv]

        qk = np.einsum("bhid,bhjd->bhij", q, k) * self.scale

        if mask is not None:
            qk = qk + mask

        attention_weights = nn.softmax(qk)

        out = np.einsum("bhij,bhjd->bhid", attention_weights, v)
        out = np.reshape(out, (x.shape[0], -1, self.num_heads * self.head_dim))

        out = nn.Dense(x.shape[-1])(out)
        out = self.dropout(out, deterministic=not train)

        return out

class PositionalEncoding(nn.Module):
    max_len: int
    dropout_rate: float

    def setup(self):
        self.positional_encoding = self.param("positional_encoding", nn.initializers.zeros, (1, self.max_len, 1))
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, train=True):
        pos_enc = self.positional_encoding[:, :x.shape[1], :]
        x = x + pos_enc
        return self.dropout(x, deterministic=not train)

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
