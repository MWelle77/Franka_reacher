import torch
from torch.autograd import Variable


class Normalization(torch.nn.Module):
    
    def __init__(self, n_features):
        super(Normalization, self).__init__()
        self.register_buffer('mean', torch.zeros(n_features))
        self.register_buffer('std', torch.ones(n_features))
        
    def forward(self, x):
        y = (x - Variable(self.mean)) / (Variable(self.std) + 1e-9)
        return y.clamp(-5, 5)


class NormalizationInverse(torch.nn.Module):
    
    def __init__(self, n_features):
        super(NormalizationInverse, self).__init__()
        self.register_buffer('mean', torch.zeros(n_features))
        self.register_buffer('std', torch.ones(n_features))
        
    def forward(self, x):
        return x * Variable(self.std) + Variable(self.mean)
