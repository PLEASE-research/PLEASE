import torch
from torch import nn, einsum
from einops import rearrange
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class DAFusion(nn.Module):
    def __init__(self, global_dim, local_dim, hidden_dim, h_out, dropout=0.2):
        super(DAFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.h_out = h_out
        self.global_fc = nn.Linear(global_dim, hidden_dim * 3)
        self.local_fc = nn.Linear(local_dim, hidden_dim * 3)
        self.dim_head = (hidden_dim*3) // h_out
        self.heads = h_out

        self.q_proj = nn.Linear(hidden_dim * 3, self.heads * self.dim_head, bias=False)
        self.k_proj = nn.Linear(hidden_dim * 3, self.heads * self.dim_head, bias=False)

        self.attention_bias = nn.Parameter(torch.Tensor(1, self.heads, 1, 1))
        self.transformer = Transformer(dim=hidden_dim * 3, depth=1, heads=3, dim_head=32, mlp_dim=128, dropout=dropout)
         # Pooling and normalization
        self.pooling = nn.AvgPool1d(3, stride=3) if h_out > 1 else None
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.final_fc = nn.Linear(hidden_dim, 1)


    def forward(self, global_feat, local_feat):
        global_proj = torch.relu(self.global_fc(global_feat))
        local_proj = torch.relu(self.local_fc(local_feat))
        global_att = self.transformer(global_proj)
        local_att = self.transformer(local_proj)
        B, V, _ = global_att.shape
        Q = local_att.size(1)
        q = self.q_proj(global_att).view(B, V, self.heads, self.dim_head).transpose(1, 2)
        k = self.k_proj(local_att).view(B, Q, self.heads, self.dim_head).transpose(1, 2)
        att_scores = (torch.matmul(q, k.transpose(-2, -1)) / (self.dim_head ** 0.5))  + self.attention_bias
        mask=(global_feat.abs().sum(dim=2) ==0)
        mask = mask.unsqueeze(1).unsqueeze(3).expand(-1, self.heads, -1, Q)
        att_scores = att_scores.masked_fill(mask, float(0))
        logits = torch.einsum('bvd,bhvq,bqd->bhd', global_att, att_scores, local_att)
        logits = logits.sum(dim=1)
        logits = self.pooling(logits.unsqueeze(1)).squeeze(1) * 3
        logits = self.bn(logits)
        logits = self.final_fc(logits)

        return logits, att_scores