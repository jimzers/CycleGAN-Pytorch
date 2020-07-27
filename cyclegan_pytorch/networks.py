import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, channels=256, alpha=0.01):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)

        self.bnorm1 = nn.BatchNorm2d(channels)
        self.bnorm2 = nn.BatchNorm2d(channels)

        self.lrelu1 = nn.LeakyReLU(alpha)

    def forward(self, _input):
        x = self.conv1(_input)
        # todo: try relu before batch norm
        x = self.bnorm1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.bnorm2(x)

        res = x + _input
        return res


class Generator(nn.Module):
    """
    9 block resnet generator as described in the CycleGAN architecture
    """
    def __init__(self, channels=3, img_height=256, img_width=256, alpha=0.01):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)

        self.bnorm1 = nn.BatchNorm2d(64)
        self.bnorm2 = nn.BatchNorm2d(128)
        self.bnorm3 = nn.BatchNorm2d(256)

        self.rblock1 = ResNetBlock(256, alpha)
        self.rblock2 = ResNetBlock(256, alpha)
        self.rblock3 = ResNetBlock(256, alpha)
        self.rblock4 = ResNetBlock(256, alpha)
        self.rblock5 = ResNetBlock(256, alpha)
        self.rblock6 = ResNetBlock(256, alpha)
        self.rblock7 = ResNetBlock(256, alpha)
        self.rblock8 = ResNetBlock(256, alpha)
        self.rblock9 = ResNetBlock(256, alpha)

        self.conv_trans1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1,
                                              output_padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                                              output_padding=1)
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3)

        self.bnorm4 = nn.BatchNorm2d(128)
        self.bnorm5 = nn.BatchNorm2d(64)

    def forward(self, _input):
        x = self.conv1(_input)
        x = F.leaky_relu(self.bnorm1(x), negative_slope=0.01)

        x = self.conv2(x)
        x = F.leaky_relu(self.bnorm2(x), negative_slope=0.01)

        x = self.conv3(x)
        x = F.leaky_relu(self.bnorm3(x), negative_slope=0.01)

        x = self.rblock1(x)
        x = self.rblock2(x)
        x = self.rblock3(x)
        x = self.rblock4(x)
        x = self.rblock5(x)
        x = self.rblock6(x)
        x = self.rblock7(x)
        x = self.rblock8(x)
        x = self.rblock9(x)

        x = self.conv_trans1(x)
        x = F.leaky_relu(self.bnorm4(x), negative_slope=0.01)

        x = self.conv_trans2(x)
        x = F.leaky_relu(self.bnorm5(x), negative_slope=0.01)

        x = self.final_conv(x)

        res = torch.tanh(x)

        return res


class Discriminator(nn.Module):
    """
    pix2pix discriminator
    """

    def __init__(self, channels=3, img_height=256, img_width=256):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1)

        self.bnorm1 = nn.BatchNorm2d(64)
        self.bnorm2 = nn.BatchNorm2d(128)
        self.bnorm3 = nn.BatchNorm2d(256)
        self.bnorm4 = nn.BatchNorm2d(512)

    def forward(self, _input):
        x = self.conv1(_input)
        x = F.leaky_relu(self.bnorm1(x), negative_slope=0.01)

        x = self.conv2(x)
        x = F.leaky_relu(self.bnorm2(x), negative_slope=0.01)

        x = self.conv3(x)
        x = F.leaky_relu(self.bnorm3(x), negative_slope=0.01)

        x = self.conv4(x)
        x = F.leaky_relu(self.bnorm4(x), negative_slope=0.01)

        res = self.conv5(x)
        return res
