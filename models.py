import torchvision
import torch.nn.functional as F
import torch.nn as nn

class ModifiedAlexNet(nn.Module):
    def __init__(self):
        super(ModifiedAlexNet, self).__init__()
        alexnet = torchvision.models.alexnet(pretrained=False)
        alexnet.features[8] = nn.Conv2d(384, 512, 3)
        self.features = alexnet.features[:10]
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = F.softmax(self.fc(x), dim=1)

        return x
