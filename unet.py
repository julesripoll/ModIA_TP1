from re import U
import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_channels, out_channels):
    # returns a block compsed of two Convolution layers with ReLU activation function
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU()
    )   

class DownSampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block=double_conv(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x=F.relu(self.conv_block(x))
        x_skip =x
        x=self.maxpool(x)
        out = x 

        return out , x_skip

class UpSampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block=double_conv(in_channels, out_channels)
        self.upsample = nn.Upsample(scale_factor=2) # use nn.Upsample

    def forward(self, x, x_skip):
        x = self.upsample(x)
        x = torch.cat([x, x_skip], dim=1) # concatenates x and x_skip. on aura donc comme dim de couch de conv 256+128, etc...
        x= F.relu(self.conv_block(x))
        return x
    

class UNet(nn.Module):

    def __init__(self):
        super().__init__()
                
        self.downsample_block_1 = DownSampleBlock(in_channels=1,out_channels=32)
        self.downsample_block_2 = DownSampleBlock(in_channels=32,out_channels=64)
        self.downsample_block_3 = DownSampleBlock(in_channels=64,out_channels=128)
        self.middle_conv_block = double_conv(128, 256)        

            
        self.upsample_block_3 = UpSampleBlock(384,128)
        self.upsample_block_2 = UpSampleBlock(192,64)
        self.upsample_block_1 = UpSampleBlock(96,32)
        
        self.last_conv = nn.Conv2d(32, 3, 1)
        
        
    def forward(self, x):
        x, x_skip1 = self.downsample_block_1(x)
        x, x_skip2 = self.downsample_block_2(x)
        x, x_skip3 = self.downsample_block_3(x)
        
        x = self.middle_conv_block(x)
        
        x = self.upsample_block_3(x,x_skip3)
        x = self.upsample_block_2(x,x_skip2)
        x = self.upsample_block_1(x,x_skip1)
        
        out = self.last_conv(x)
        
        return out

        
if __name__=='__main__':
    x = torch.rand(16,1,224,224)
    net = UNet()
    y = net(x)
    assert y.shape == (16,3,224,224)
    print('Shapes OK')
