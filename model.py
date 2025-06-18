import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as packer, pad_packed_sequence as padder

# ----------------------------------------------------------------------------------------------------------------------

class GRUOutput(nn.Module):
    """
    GRU wrapper that only returns output (rather than hidden state)
    Used for tidier code in DeepGRU
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gru = nn.GRU(*args, **kwargs)

    def forward(self, x):
        output, _ = self.gru(x)
        return output


class DeepGRU(nn.Module):
    def __init__(self, num_features, num_classes, layer_size=64, with_context=True):
        super(DeepGRU, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.with_context = with_context

        if with_context:
            # Preceeding svara encoder
            self.prec_encoder = nn.Sequential(
                GRUOutput(self.num_features, layer_size, 2, batch_first=True),
                GRUOutput(layer_size, layer_size // 2, 2, batch_first=True),
                nn.GRU(layer_size // 2, layer_size // 4, 1, batch_first=True)
            )
        
        # Current svara encoder
        self.curr_encoder = nn.Sequential(
            GRUOutput(self.num_features, layer_size, 2, batch_first=True),
            GRUOutput(layer_size, layer_size // 2, 2, batch_first=True),
            nn.GRU(layer_size // 2, layer_size // 4, 1, batch_first=True)
        )
        
        if with_context:
            # Succeeding svara encoder
            self.succ_encoder = nn.Sequential(
                GRUOutput(self.num_features, layer_size, 2, batch_first=True),
                GRUOutput(layer_size, layer_size // 2, 2, batch_first=True),
                nn.GRU(layer_size // 2, layer_size // 4, 1, batch_first=True)
            )

        # Attention
        self.attention = Attention(layer_size // 4)

        if self.with_context:
            attn_size = (layer_size // 2)*3
        else:
            attn_size = (layer_size // 2)

        # Projection Head for Contrastive Learning
        self.projection_head = nn.Sequential(
            nn.Linear(attn_size, attn_size // 2),
            nn.ReLU(),
            nn.Linear(attn_size // 2, attn_size // 2)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(attn_size),
            nn.Dropout(0.5),
            nn.Linear(attn_size, attn_size),
            nn.ReLU(),
            nn.BatchNorm1d(attn_size),
            nn.Dropout(0.5),
            nn.Linear(attn_size, num_classes)
        )

    def embed(self, x_prec_padded, x_curr_padded, x_succ_padded, x_prec_lengths, x_curr_lengths, x_succ_lengths):
        """Extracts embeddings from the GRU layers."""
        
        x_curr_packed = packer(x_curr_padded, x_curr_lengths.to('cpu'), batch_first=True, enforce_sorted=False)        
        output_curr, hidden_curr = self.curr_encoder(x_curr_packed)
        output_curr_padded, _ = padder(output_curr, batch_first=True)
        attn_curr = self.attention(output_curr_padded, hidden_curr[-1:])
        
        if self.with_context:
            x_prec_packed = packer(x_prec_padded, x_prec_lengths.to('cpu'), batch_first=True, enforce_sorted=False)
            output_prec, hidden_prec = self.prec_encoder(x_prec_packed)
            output_prec_padded, _ = padder(output_prec, batch_first=True)    
            attn_prec = self.attention(output_prec_padded, hidden_curr[-1:])

            x_succ_packed = packer(x_succ_padded, x_succ_lengths.to('cpu'), batch_first=True, enforce_sorted=False)
            output_succ, hidden_succ = self.succ_encoder(x_succ_packed)
            output_succ_padded, _ = padder(output_succ, batch_first=True)
            attn_succ = self.attention(output_succ_padded, hidden_curr[-1:])
            attn_output = torch.cat([attn_prec, attn_curr, attn_succ], dim=-1)
        else:
            attn_output = attn_curr
                    
        return attn_output

    def forward(self, x_prec_padded, x_curr_padded, x_succ_padded, x_prec_lengths, x_curr_lengths, x_succ_lengths, return_embeddings=False, contrastive=False):
        attn_output = self.embed(x_prec_padded, x_curr_padded, x_succ_padded, x_prec_lengths, x_curr_lengths, x_succ_lengths)

        if return_embeddings:
            return attn_output  # Used for evaluation or feature extraction
        
        if contrastive:
            # Normalize embeddings for contrastive loss
            projection = self.projection_head(attn_output)
            return F.normalize(projection, dim=1)  # Normalized embeddings for contrastive loss
        
        return self.classifier(attn_output)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, attention_dim):
        super(Attention, self).__init__()
        self.w = nn.Linear(attention_dim, attention_dim, bias=False)
        self.gru = nn.GRU(attention_dim, attention_dim, 1, batch_first=True)

    def forward(self, input_padded, hidden):
        e = torch.bmm(self.w(input_padded), hidden.permute(1, 2, 0))
        context = torch.bmm(input_padded.permute(0, 2, 1), e.softmax(dim=1))
        context = context.permute(0, 2, 1)

        # Compute auxiliary context, and concatenate
        aux_context, _ = self.gru(context, hidden)
        output = torch.cat([aux_context, context], 2).squeeze(1)

        return output
