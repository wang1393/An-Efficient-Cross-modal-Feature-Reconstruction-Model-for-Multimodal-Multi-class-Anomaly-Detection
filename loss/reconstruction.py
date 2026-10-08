"""Multi-scale L2 feature reconstruction loss."""
import torch.nn as nn
from . import LOSS

@LOSS.register_module
class L2Loss(nn.Module):
    def __init__(self, lam=1):
        super(L2Loss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')
        self.lam = lam

    def forward(self, input1, input2):
        input1 = input1 if isinstance(input1, list) else [input1]
        input2 = input2 if isinstance(input2, list) else [input2]
        loss = 0
        for in1, in2 in zip(input1, input2):
            loss += self.loss(in1, in2) * self.lam
        return loss
