from torchvision.datasets import STL10
from torchvision import transforms
from .datapath import get_data_folder


def get_dataset():
    dpath = get_data_folder()
    dataset = STL10(dpath + '/STL10', split='unlabeled',
                    transform=transforms.Compose([transforms.Resize(48),
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
