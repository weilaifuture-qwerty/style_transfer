import torch
import torch.nn as nn

device = torch.device('mps')
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.reflection_pad(x)
        y = self.conv(x)
        y = self.batch_norm(y)
        y = self.relu(y)
        return y
    
class ResBlock(nn.Module):
    def __init__(self, filters):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = filters, out_channels = filters, kernel_size = 3, stride = 1, padding = 1, device = device)
        self.batch_norm1 = nn.BatchNorm2d(filters, device = device)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = filters, out_channels = filters, kernel_size = 3, stride = 1, padding = 1, device = device)
        self.batch_norm2 = nn.BatchNorm2d(filters, device = device)

    def forward(self, x):
        y = self.conv1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.batch_norm2(y)
        y = y + x
        return y
    
class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(UpSampling, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = 1, device = device)        
        self.batch_norm = nn.BatchNorm2d(out_channels, device = device)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        y = self.upsample(x)
        y = self.reflection_pad(y)
        y = self.conv1(y)
        y = self.batch_norm(y)
        y = self.relu(y)
        return y
    

class ImageTransformNet(nn.Module):
    def __init__(self):
        super(ImageTransformNet, self).__init__()
        self.conv1 = ConvLayer(3, 32, 9, 1, 4)
        self.conv2 = ConvLayer(32, 64, 3, 2, 1)
        self.conv3 = ConvLayer(64, 128, 3, 2, 1)
        self.res1 = ResBlock(128)
        self.res2 = ResBlock(128)
        self.res3 = ResBlock(128)
        self.res4 = ResBlock(128)
        self.res5 = ResBlock(128)
        self.up1 = UpSampling(128, 64, 3, 1)
        self.up2 = UpSampling(64, 32, 3, 1)
        self.conv4 = ConvLayer(32, 3, 9, 1, 4)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.up1(y)
        y = self.up2(y)
        y = self.conv4(y)
        return y