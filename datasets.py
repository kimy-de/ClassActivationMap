import torch
import torchvision
import torchvision.transforms as transforms

class Datasets:

    def __init__(self, img_size, path, name):

        self.img_size = img_size
        self.path = path
        self.name = name

    def stl10(self):

         transform = transforms.Compose(
             [transforms.Resize(self.img_size), transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

         trainset = torchvision.datasets.STL10(root=self.path + 'data', split='train', download=True, transform=transform)
         trainloader = torch.utils.data.DataLoader(trainset, batch_size=40, shuffle=True)

         return trainset, trainloader

    def create_dataset(self):

        if self.name == 'stl10':

            return self.stl10()

        elif self.name == 'custom':

            print("custom")

