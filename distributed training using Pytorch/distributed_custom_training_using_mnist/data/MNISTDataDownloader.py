import torchvision.datasets as dsets
import utils.HelperFunctions as util


def getTrainDataset(image_size):
    composed = util.getTransformFn(image_size)
    dataset = dsets.MNIST(root='./data', train=True, transform=composed, download=True)
    return dataset


def getValidationDataset(image_size):
    composed = util.getTransformFn(image_size)
    dataset = dsets.MNIST(root='./data', train=False, transform=composed, download=True)
    return dataset