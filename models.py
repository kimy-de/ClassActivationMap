import torchvision
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.mean([2, 3])
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)

        return x

def m_alexnet():
    model = torchvision.models.alexnet(pretrained=False)
    model.features[8] = nn.Conv2d(384, 512, 3)
    model = nn.Sequential(OrderedDict([('features', model.features[:10]), ('classifier', Classifier())]))
    return model
