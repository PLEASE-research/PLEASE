import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence



class plaCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super(plaCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.W_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_q = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_k = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_v = nn.Parameter(torch.Tensor(input_size, hidden_size))

        if self.bias:
            self.B_i = nn.Parameter(torch.zeros(hidden_size))
            self.B_f = nn.Parameter(torch.zeros(hidden_size))
            self.B_o = nn.Parameter(torch.zeros(hidden_size))
            self.B_q = nn.Parameter(torch.zeros(hidden_size))
            self.B_k = nn.Parameter(torch.zeros(hidden_size))
            self.B_v = nn.Parameter(torch.zeros(hidden_size))

        self.reset_parameters()
        self.norm = nn.LayerNorm(input_size)

    def reset_parameters(self):
        for weight in self.parameters():
            if weight.dim() > 1:
                nn.init.orthogonal_(weight)
            else:
                nn.init.zeros_(weight)


    def forward(self, x, state):
        C, n, m = state

        x = self.norm(x)

        i_tilda = torch.matmul(x, self.W_i) + self.B_i
        f_tilda = torch.matmul(x, self.W_f) + self.B_f
        o_tilda = torch.matmul(x, self.W_o) + self.B_o

        # 限制输入范围防止梯度爆炸
        i_tilda = torch.clamp(i_tilda, min=-100.0, max=100.0)
        f_tilda = torch.clamp(f_tilda, min=-100.0, max=100.0)
        o_tilda = torch.clamp(o_tilda, min=-100.0, max=100.0)

        q_t = torch.matmul(x, self.W_q) + self.B_q
        k_t = torch.matmul(x, self.W_k) / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32)) + self.B_k
        v_t = torch.matmul(x, self.W_v) + self.B_v

        i_t = torch.exp(i_tilda)
        f_t = torch.sigmoid(f_tilda)
        o_t = torch.sigmoid(o_tilda)

        m_t = torch.max(
            torch.log(f_t.clamp(min=1e-8)) + m,
            torch.log(i_t.clamp(min=1e-8))
        )
        i_prime = torch.exp((i_t - m_t).clamp(max=20.0))

        C_t = f_t.unsqueeze(-1) * C + i_prime.unsqueeze(-1) * torch.einsum("bi,bk->bik", v_t, k_t)
        n_t = f_t * n + i_prime * k_t

        normalize_inner = torch.einsum("bh,bh->b", n_t, q_t)
        divisor = torch.max(torch.abs(normalize_inner),torch.ones_like(normalize_inner))
        h_tilda = torch.einsum("bik,bk->bi", C_t, q_t) / divisor.unsqueeze(-1)
        h_t = o_t * h_tilda

        return h_t, (C_t, n_t, m_t)

    def init_hidden(self, batch_size: int, **kwargs):
        return (
            torch.zeros(batch_size, self.hidden_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
        )



class PLA(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2,
                 bias: bool = True, batch_first: bool = False, dropout = 0.2, bidirectional: bool = False) -> None:
        super(PLA, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.layers = nn.ModuleList([
            plaCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                bias
            ) for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fusion_mlp = nn.Linear(hidden_size * num_layers, hidden_size)

    def forward(self, x, states=None):
        packed_input = isinstance(x, PackedSequence)
        if packed_input:
            x, lengths = pad_packed_sequence(x, batch_first=self.batch_first)

        batch_size = x.size(0) if self.batch_first else x.size(1)
        seq_len = x.size(1) if self.batch_first else x.size(0)

        if states is None:
            states = self.layers[0].init_hidden(batch_size, device=x.device)

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :] if self.batch_first else x[t]
            layer_outputs = []

            for i, layer in enumerate(self.layers):
                h_t, states = layer(x_t, states)
                x_t = h_t
                layer_outputs.append(h_t)
            fusion_input = torch.cat(layer_outputs, dim=-1)
            fusion_input = self.dropout(fusion_input)
            fused_h_t = self.fusion_mlp(fusion_input)

            outputs.append(fused_h_t+layer_outputs[-1])

        outputs = torch.stack(outputs, dim=1 if self.batch_first else 0)

        if packed_input:
            outputs = pack_padded_sequence(outputs, lengths, batch_first=self.batch_first, enforce_sorted=False)

        return outputs, states[-1][1]