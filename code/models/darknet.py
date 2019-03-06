import torch.nn as nn
import torchvision.models as m
import math
from utils.utils import *
from utils.switch_norm_torch import SwitchNorm2d


class DarkNet(nn.Module):
    def __init__(self, module_list, num_classes, init_weights=True):
        super(DarkNet, self).__init__()
        self.module_list = module_list
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, num_classes, 1, 1, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(True),
            nn.AvgPool2d(7, 1),

        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        print(x.shape)
        for module in self.module_list:
            x = module(x)
            print(x.shape)
        print(x.shape)
        x = self.classifier(x)
        x = x.squeeze()
        print(x.shape)
        return x

    def _initialize_weights(self):
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


def _conv2d_bn_relu(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        # SwitchNorm2d(out_channels),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True),
        # nn.Dropout(),
    )


def _maxpool():
    return nn.Sequential(
        nn.MaxPool2d(2, 2)
    )


def _make_module_list():
    modules = nn.ModuleList()
    modules.append(_conv2d_bn_relu(3, 32, 3, 1, 1))
    modules.append(_maxpool())

    modules.append(_conv2d_bn_relu(32, 64, 3, 1, 1))
    modules.append(_maxpool())

    modules.append(_conv2d_bn_relu(64, 128, 3, 1, 1))
    modules.append(_conv2d_bn_relu(128, 64, 1, 1, 0))
    modules.append(_conv2d_bn_relu(64, 128, 3, 1, 1))
    modules.append(_maxpool())

    modules.append(_conv2d_bn_relu(128, 256, 3, 1, 1))
    modules.append(_conv2d_bn_relu(256, 128, 3, 1, 0))
    modules.append(_conv2d_bn_relu(128, 256, 3, 1, 1))
    modules.append(_maxpool())

    modules.append(_conv2d_bn_relu(256, 512, 3, 1, 1))
    modules.append(_conv2d_bn_relu(512, 256, 3, 1, 1))
    modules.append(_conv2d_bn_relu(256, 512, 3, 1, 1))
    modules.append(_conv2d_bn_relu(512, 256, 3, 1, 1))
    modules.append(_conv2d_bn_relu(256, 512, 3, 1, 1))
    modules.append(_maxpool())

    modules.append(_conv2d_bn_relu(512, 1024, 3, 1, 1))
    modules.append(_conv2d_bn_relu(1024, 512, 3, 1, 1))
    modules.append(_conv2d_bn_relu(512, 1024, 3, 1, 1))
    modules.append(_conv2d_bn_relu(1024, 512, 3, 1, 1))
    modules.append(_conv2d_bn_relu(512, 1024, 3, 1, 1))

    return modules


def darknet(num_classes, init_weights=True):
    return DarkNet(_make_module_list(), num_classes=num_classes, init_weights=init_weights)


def test_net():
    import torch
    x = torch.randn((2, 3, 224, 224))
    model = darknet(5)
    model(x)
    get_number_of_param(model)


if __name__ == '__main__':
    test_net()
    # model = m.resnet101(pretrained=False)
    # get_number_of_param(model)
