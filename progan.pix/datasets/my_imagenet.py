from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
from .datapath import get_data_folder


def get_dataset():
    dpath = get_data_folder()
    traindir = os.path.join(dpath, 'ImageNet/train')
    dataset = ImageFolder(
        traindir,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)),
        ]))
    return dataset


if __name__ == '__main__':
    dataset = get_dataset()
    print(len(dataset))
    d, l = dataset[100]
    print(d.shape, d.min(), d.max())
