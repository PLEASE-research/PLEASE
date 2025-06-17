import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from PLA import PLA
from DAIF import DAFusion


class PLEASE(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, output_dim, dropout, device):
        super(PLEASE, self).__init__()

        self.PLA = PLA(input_size=embed_dim,hidden_size=hidden_dim,num_layers=num_layers,bias=True,batch_first=True,dropout=dropout
        )
        self.device = device
        self.DAFusion =DAFusion(global_dim=embed_dim, local_dim=hidden_dim, hidden_dim=output_dim, h_out=2, dropout=dropout).to(self.device)

    def forward(self, code_tensor):
        sent_lengths = [len(code) for code in code_tensor]
        sent_lengths = torch.tensor(sent_lengths).to(self.device)
        code_tensor = pad_sequence(code_tensor, batch_first=True)
        sent_lengths, sent_perm_idx = sent_lengths.sort(dim=0, descending=True)
        code_tensor = code_tensor[sent_perm_idx]
        packed_sents = pack_padded_sequence(code_tensor, lengths=sent_lengths.tolist(), batch_first=True)
        line_level_contexts, _ = self.mlstm(packed_sents)
        line_level_contexts, _ = pad_packed_sequence(line_level_contexts, batch_first=True)
        _, sent_unperm_idx = sent_perm_idx.sort(dim=0, descending=False)
        code_tensor = code_tensor[sent_unperm_idx]
        line_level_contexts = line_level_contexts[sent_unperm_idx]

        file_level_embeds, sent_att = self.CrossChannelFusion( code_tensor,line_level_contexts)


        sent_att = sent_att.sum(dim=1)
        sent_att_weights = [item.diag() for item in sent_att]
        sent_att_weights = torch.stack(sent_att_weights, dim=0)
        sent_att_weights = sent_att_weights / torch.sum(sent_att_weights, dim=1, keepdim=True)

        return file_level_embeds, sent_att_weights