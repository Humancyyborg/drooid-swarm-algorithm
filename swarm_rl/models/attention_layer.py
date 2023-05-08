import torch
from torch import nn
import torch.nn.functional as F

"""
Implementation of Attention Block from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
"""


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        size_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(size_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(size_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(size_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(size_b, len_q, -1)
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn


class OneHeadAttention(nn.Module):
    """ One-Head Attention module """

    def __init__(self, d_model):
        super().__init__()

        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)

        self.fc = nn.Linear(d_model, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.d_model = d_model

    def forward(self, q, k, v):
        residual = q

        # Pass through the pre-attention projection: b x lq x dv
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        # Compute attention weights using queries and keys
        attn = torch.matmul(q / (self.d_model ** 0.5), k.transpose(-1, -2))
        # attn /= torch.sqrt(self.d_model)
        attn = F.softmax(attn, dim=-1)
        q = torch.matmul(attn, v)

        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn
