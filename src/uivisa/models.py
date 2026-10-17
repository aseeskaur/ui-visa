import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# U-Net used to produce the initial seeds.
# Copied unchanged from dsa_unet_train_full_aug_with_seed.py.
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout=0.2):
        super(UNet, self).__init__()

        self.enc1 = DoubleConv(in_channels, 64, dropout)
        self.enc2 = DoubleConv(64, 128, dropout)
        self.enc3 = DoubleConv(128, 256, dropout)
        self.enc4 = DoubleConv(256, 512, dropout)

        self.bottleneck = DoubleConv(512, 1024, dropout)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512, dropout)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256, dropout)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128, dropout)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64, dropout)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        x = F.max_pool2d(enc1, 2)

        enc2 = self.enc2(x)
        x = F.max_pool2d(enc2, 2)

        enc3 = self.enc3(x)
        x = F.max_pool2d(enc3, 2)

        enc4 = self.enc4(x)
        x = F.max_pool2d(enc4, 2)

        x = self.bottleneck(x)

        x = self.upconv4(x)
        x = torch.cat([enc4, x], dim=1)
        x = self.dec4(x)

        x = self.upconv3(x)
        x = torch.cat([enc3, x], dim=1)
        x = self.dec3(x)

        x = self.upconv2(x)
        x = torch.cat([enc2, x], dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        x = torch.cat([enc1, x], dim=1)
        x = self.dec1(x)

        x = self.final_conv(x)
        return x


# ---------------------------------------------------------------------------
# Patch CNN used for region growing: 80x80 input patch -> 3x3 output.
# Copied unchanged from rgc_train_aug_resamp_new_fixing_plots.py / hybrid_test.py.
# Class and attribute names are kept so state_dict keys stay the same.
# ---------------------------------------------------------------------------

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv1 = double_conv(1, 64)
        self.conv2 = double_conv(64, 128)
        self.conv3 = double_conv(128, 256)
        self.conv4 = double_conv(256, 512)
        self.conv5 = double_conv(512, 1024)
        self.conv6 = nn.Conv2d(1024, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, image):
        x1 = self.conv1(image)
        x2 = self.maxpool(x1)
        x3 = self.conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.conv4(x6)
        x8 = self.maxpool2(x7)
        x9 = self.conv5(x8)
        x10 = self.conv6(x9)
        return x10


PatchCNN = net
