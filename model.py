"""
Transformer text classifier (PyTorch).
Includes attention, encoder blocks, and a 2-class classifier head.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

def my_scaled_dot_product_attention(Q, K, V):
    # Q has shape (bs, nh, T, d_key)
    # K has shape (bs, nh, T, d_key)
    # V has shape (bs, nh, T, d_val)
    # Y has shape (bs, nh, T, d_val)

    d_key = K.shape[-1]
    A_tilde = Q @ K.transpose(-2,-1) / math.sqrt(d_key)    # (bs, nh, T, T)
    A = F.softmax(A_tilde, dim=-1)                         # (bs, nh, T, T)
    Y = A @ V                                              # (bs, nh, T, d_val)
    return Y

class EfficientMultiHeadAttention(nn.Module):

    def __init__(self, d, num_heads):
        super().__init__()

        assert d % num_heads == 0
        d_key = d // num_heads
        d_val = d // num_heads

        self.linear_q = nn.Linear(d, num_heads*d_key, bias=False)
        self.linear_k = nn.Linear(d, num_heads*d_key, bias=False)
        self.linear_v = nn.Linear(d, num_heads*d_val, bias=False)
        self.linear_out = nn.Linear(d, d, bias=False)

        self.nh = num_heads
        self.dk = d_key
        self.dv = d_val

    def forward(self, X):
        # X has shape (bs, T, d)
        bs = X.shape[0]
        T = X.shape[1]

        Q = self.linear_q(X)    # (bs, T, nh*dk)
        K = self.linear_k(X)    # (bs, T, nh*dk)
        V = self.linear_v(X)    # (bs, T, nh*dv)

        Q = Q.view(bs, T, self.nh, self.dk)   # (bs, T, nh, dk)
        K = K.view(bs, T, self.nh, self.dk)   # (bs, T, nh, dk)
        V = V.view(bs, T, self.nh, self.dv)   # (bs, T, nh, dv)

        Q = Q.transpose(1,2)    # (bs, nh, T, dk)
        K = K.transpose(1,2)    # (bs, nh, T, dk)
        V = V.transpose(1,2)    # (bs, nh, T, dv)

        Y = my_scaled_dot_product_attention(Q,K,V)    # (bs, nh, T, dv)

        Y = Y.transpose(1,2).contiguous()     # (bs, T, nh, dv)
        Y = Y.view(bs, T, self.nh*self.dv)    # (bs, T, nh*dv)

        Y = self.linear_out(Y)                # (bs, T, d)
        return Y

class MLP(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.linear_1 = nn.Linear(d, 4*d, bias=False)
        self.linear_2 = nn.Linear(4*d, d, bias=False)

    def forward(self, X):
        # X has shape (bs, T, d)
        Y = self.linear_1(X)        # (bs, T, 4d)
        Y_hat = F.relu(Y)           # (bs, T, 4d)
        Z = self.linear_2(Y_hat)    # (bs, T, d)
        return Z

class TransformerBlock(nn.Module):

    def __init__(self, d, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = EfficientMultiHeadAttention(d, num_heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = MLP(d)

    def forward(self, X):
        # X has shape (bs, T, d)
        X1 = self.ln1(X)
        Y = self.attn(X1)    # (bs, T, d)
        X2 = X + Y           # residual
        X3 = self.ln2(X2)
        Z = self.mlp(X3)     # (bs, T, d)
        output = X2 + Z      # residual
        return output

class MyTransformer(nn.Module):

    def __init__(self, vocab_size, d, num_heads, num_layers, T):
        super().__init__()

        self.word_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(T, d)

        blocks = []
        for i in range(num_layers):
            blocks.append(TransformerBlock(d, num_heads))
        self.blocks = nn.ModuleList(blocks)

        self.last_linear = nn.Linear(d, 2, bias=False)    # last linear layer to 2 classes

    def forward(self, X):
        # X has shape (bs, T)
        bs = X.shape[0]
        T = X.shape[1]

        pos = torch.arange(T, device = X.device)     # (T,)
        pos = pos.expand(bs, T)                      # (bs, T)

        Xw = self.word_emb(X)                        # (bs, T, d)
        Xp = self.pos_emb(pos)                       # (bs, T, d)
        X = Xw + Xp                                  # (bs, T, d)

        for block in self.blocks:
            X = block(X)                             # (bs, T, d)

        # average over time
        X_avg = X.mean(dim=1)                        # (bs, d)

        scores = self.last_linear(X_avg)
        return scores
