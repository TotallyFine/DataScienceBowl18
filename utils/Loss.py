# coding:utf-8
from torch import nn
class Loss(nn.Module):
    def __init__(self, name):
        super(Loss, self).__init__()
        assert isinstance(name, str)
        self.name = name
    
        
    def forward(self, inputs, targets):
        # batch size
        num = targets.size(0)
        m1 = inputs.view(num, -1)
        m2 = inputs.view(num, -1)
        intersection = m1.mul(m2)
        x = intersection.sum(1).add(1)
        y = m1.sum(1).add(m2.sum(1)).add(1)
        score = x.div(y).mul(2.0)
        score = score.sum().div(num).neg().add(1)
        # intersection = (m1 * m2)
        # score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
        # score = 1 - score.sum()/num
        return score
