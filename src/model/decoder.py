import torch
from torch import nn
from src.model.multi_layer_gru_cell import MultiLayerGRUCell
from src.constants import SOS_INDEX

class Decoder(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, weight_tying):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.rnn_cell = MultiLayerGRUCell(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, embed_size)
        )
        self.generator = nn.Linear(embed_size, vocab_size)
        if weight_tying:
            self.generator.weight = self.embedding.weight

    def forward(self, hidden, trg):
        max_len = trg.size(1)
        logit = []
        for i in range(max_len):
            hidden, token_logit = self.step(hidden, trg[:, i])
            logit.append(token_logit)
        logit = torch.stack(logit, dim=1)
        return logit

    def step(self, hidden, token):
        token_embedding = self.embedding(token.unsqueeze(0)).squeeze(0)
        hidden = self.rnn_cell(token_embedding, hidden)
        top_hidden = hidden[-1]
        output = self.output_projection(top_hidden)
        token_logit = self.generator(output)
        return hidden, token_logit

    def decode(self, hidden, max_len):
        batch_size = hidden.size(1)
        token = torch.tensor([SOS_INDEX] * batch_size, dtype=torch.long, device=hidden.device)
        logit = []
        for i in range(max_len):
            hidden, token_logit = self.step(hidden, token)
            token = token_logit.argmax(dim=-1)
            logit.append(token_logit)
        logit = torch.stack(logit, dim=1)
        return logit