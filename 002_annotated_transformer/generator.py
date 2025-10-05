from torch.nn.functional import log_softmax
import torch.nn as nn

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)
