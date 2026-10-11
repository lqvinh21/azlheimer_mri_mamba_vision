from torchvision import transforms

from ..config import IMAGENET_MEAN, IMAGENET_STD


def make_train_transform(img_size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(7),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_eval_transform(img_size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
