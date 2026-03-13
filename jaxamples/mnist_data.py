from torchvision import transforms


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def get_mnist_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )
