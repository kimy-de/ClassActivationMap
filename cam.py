import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Visualize feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def cam(model, trainset, img_sample, img_size):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.features[9].register_forward_hook(get_activation('final'))
    data, label = trainset[img_sample]
    data.unsqueeze_(0)
    _ = model(data.to(device))
    act = activation['final'].squeeze()
    w = model.fc.weight
    #print(w.size())

    for idx in range(act.size(0)):
        if idx == 0:
            tmp = act[idx] * w[label][idx]
        else:
            tmp += act[idx] * w[label][idx]

    normalized_cam = tmp.detach().cpu().numpy()
    normalized_cam = (normalized_cam - np.min(normalized_cam)) / (np.max(normalized_cam) - np.min(normalized_cam))

    cam_img = cv2.resize(np.uint8(normalized_cam * 255), dsize=(img_size, img_size))
    original_img = np.uint8((data[0][0] / 2 + 0.5) * 255)

    return cam_img, original_img


def plot_cam(model, trainset, img_size, start):
    end = start + 20
    fig, axs = plt.subplots(2, (end - start + 1) // 2, figsize=(20, 5))
    fig.subplots_adjust(hspace=.01, wspace=.01)
    axs = axs.ravel()

    for i in range(start, end):
        cam_img, original_img = cam(model, trainset, i, img_size)

        axs[i - start].imshow(original_img, cmap='gray')
        axs[i - start].imshow(cam_img, cmap='jet', alpha=.5)
        axs[i - start].axis('off')

    plt.show()
    fig.savefig('cam.png')


