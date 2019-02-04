from torchvision.datasets import CIFAR10
from torchvision import transforms
from .datapath import get_data_folder


def get_dataset(dpath=None, ):
    dpath = get_data_folder()
    dataset = CIFAR10(dpath + '/CIFAR10', train=True,
                      transform=transforms.Compose([transforms.Resize(32),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        mean=(0.5, 0.5, 0.5),
                                                        std=(0.5, 0.5, 0.5)),
                                                    ]), download=True)
    return dataset


if __name__ == '__main__':
    dataset = get_dataset()
    print(len(dataset))
    d, l = dataset[0]
    print(d.shape, d.min(), d.max())
