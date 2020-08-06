import torch
from torch import nn
import torch.nn.functional as F
from src.model.encoder import Encoder
from src.model.decoder import Decoder

class TextVAE(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, enc_dec_tying, dec_gen_tying):
        super(TextVAE, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            weight_tying=dec_gen_tying
        )
        if enc_dec_tying:
            self.decoder.embedding.weight = self.encoder.embedding.weight
        self.mean_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.std_projection = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, src, trg):
        encoding, mean, std = self.encode(src)
        logit = self.decoder(encoding, trg)
        return logit, mean, std

    def encode(self, src):
        final_states = self.encoder(src)
        mean = self.mean_projection(final_states)
        std = F.softplus(self.std_projection(final_states))
        sample = torch.randn(size=mean.size(), device=mean.device)
        encoding = mean + std * sample
        return encoding, mean, std

    def sample(self, **kwargs):
        assert ('num' in kwargs) ^ ('encoding' in kwargs)
        if 'num' in kwargs:
            encoding = torch.randn(size=(self.num_layers, kwargs['num'], self.hidden_size),
                                   device=self.encoder.embedding.weight.device)
        else:
            encoding = kwargs['encoding']
        logit = self.decoder.decode(encoding, 20)
        output = logit.argmax(dim=-1)
        if 'output_encoding' in kwargs and kwargs['output_encoding']:
            num = encoding.size(1)
            encoding = encoding.transpose(0, 1).reshape(num, -1).cpu().numpy()
            return output, encoding
        else:
            return output