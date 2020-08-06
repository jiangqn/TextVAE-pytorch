import torch
from torch import nn
import torch.nn.functional as F

class MultiLayerGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, bias=True):
        super(MultiLayerGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bias = bias
        self.gru_cells = nn.ModuleList([nn.GRUCell(input_size, hidden_size, bias)])
        for _ in range(num_layers - 1):
            self.gru_cells.append(nn.GRUCell(input_size, hidden_size, bias))

    def forward(self, input, states):
        """
        :param input: FloatTensor (batch_size, time_step, input_size)
        :param states: FloatTensor (num_layers, batch_size, hidden_size)
        :return output_hidden: FloatTensor (num_layers, batch_size, hidden_size)
        """
        hidden = states
        output_hidden = []
        for i, gru_cell in enumerate(self.gru_cells):
            h = gru_cell(input, hidden[i])
            output_hidden.append(h)
            input = F.dropout(h, p=self.dropout, training=self.training)
        output_hidden = torch.stack(output_hidden, dim=0)
        return output_hidden