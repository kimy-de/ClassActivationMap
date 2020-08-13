import cam
import datasets
import argparse
import models

import torch
import torch.nn as nn # loss
import torch.optim as optim # optimizer



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Class Activation Map')
    parser.add_argument('--img_size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', default=20, type=int, help='training epoch')
    parser.add_argument('--plot_start', default=100, type=int, help='plot_start')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--data_name', default='stl10', type=str, help='dataset')
    parser.add_argument('--data_path', default='./', type=str, help='dataset path')
    parser.add_argument('--model_path', default='./cam_model.pth', type=str, help='model path')
    parser.add_argument('--training', default=True, help='training model')

    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = datasets.Datasets(args.img_size, args.data_path, args.data_name)
    trainset, trainloader = data.create_dataset()

    model = models.m_alexnet()
    model = model.to(device)
    #print(model)

    if args.training == True:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        old_loss = 10

        for epoch in range(args.epoch):  # 100

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            cost = running_loss / len(trainloader)
            print('[%d] loss: %.3f' % (epoch + 1, cost))

        print('Finished Training')

        torch.save(model.state_dict(), args.model_path)

    else:
        model.load_state_dict(torch.load(args.model_path))

    create_cam = cam.plot_cam(model, trainset, args.img_size, args.plot_start)