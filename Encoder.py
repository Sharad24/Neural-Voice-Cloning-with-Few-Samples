import torch
import torch.nn as nn
import torch.nn.functional as F

import Cloning_Samples_Attention
from Cloning_Samples_Attention import *

class Attention(nn.Module):
    def __init__(self, dim, num_units):
        super(Attention, self).__init__()

        self.encoders = self._build_model(dim, num_units)
        
    def _build_model(self, dim, num_units):
        layers = []
        # for encoder, we use self-attention, which means we
        # have query_dim and key_dim with same size
        dim = dim
        for i in range(num_units):
            layer = ExtendedSequential(
                MultiHeadAttention(dim, dim, i),
                PositionWiseFFN(i))
            layers.append(layer)
            i = unit

        return nn.ModuleList(layers)

    def forward(self, inputs, conv_unit_output):
        net_inputs = inputs
        for enc in self.encoders:
            net_inputs = enc(net_inputs, net_inputs)
        return net_inputs
