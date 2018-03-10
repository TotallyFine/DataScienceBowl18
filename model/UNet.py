# coding:utf-8
import torch.nn.functional as F
from torch import nn
import torch
from .BasicModule import BasicModule

class Block(nn.Module):
    # inchannel outchannel
    def __init__(self, in_c, out_c, stride=1):
        super(Block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
        
class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super(UpBlock, self).__init__()
        # the in_c is x1's channel and out_c is x2's channel
        self.upconv =nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = Block(2*out_c, out_c)
        
    def forward(self, x1, x2):
        '''
        x1 need to upconv
        '''
        x1 = self.upconv(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        assert diffY == 0 and diffX == 0
        #x2 = F.pad(x2, (diffX // 2, int(diffX // 2)),
                      # (diffY // 2, int(diffY // 2)))
        # cat along dim=1(channel) dim=0 is batch
        x = torch.cat((x2, x1), dim=1)
        x = self.conv(x)
        return x


class UNet(BasicModule):
    '''
    a simple implementation of UNet
    input channel=3 h=resize_height w=resize_width
    output channel=n_classes h=resize_height w=resize_width
    '''
    def __init__(self, in_channel, n_classes):
        super(UNet, self).__init__()
        self.model_name = 'UNet'
        
        self.preBlock = Block(in_channel, 24)
        
        # maxpool first
        # the feature map can be used for up after downconv
        # h/2 w/2
        self.maxpool1 = nn.MaxPool2d(2)
        self.down1 = Block(24, 64)
        
        # h/4 w/4
        self.maxpool2 = nn.MaxPool2d(2)
        self.down2 = Block(64, 128)
        
        # h/8 w/8
        self.maxpool3 = nn.MaxPool2d(2)
        self.down3 = Block(128, 256)
        
        # h/16 w/16
        self.maxpool4 = nn.MaxPool2d(2)
        self.down4 = Block(256, 512)
        
        # x1=down4's output x2=down3's output
        self.up1 = UpBlock(512, 256)
        
        # x1 = up1 x2 = down2
        self.up2 = UpBlock(256, 128)
        
        # x1 = up2 x2 = down1
        self.up3 = UpBlock(128, 64)
        
        # x1 = up3, x2 = preBlock
        self.up4 = UpBlock(64, 24)
        
        # do not have dropout
        self.output = nn.Conv2d(24, n_classes, kernel_size=1)
        
    def forward(self, x):
        pre = self.preBlock(x)
        down1 = self.down1(self.maxpool1(pre))
        down2 = self.down2(self.maxpool2(down1))
        down3 = self.down3(self.maxpool3(down2))
        down4 = self.down4(self.maxpool4(down3))
        up1 = self.up1(down4, down3)
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)
        up4 = self.up4(up3, pre)
        x = self.output(up4)
        return F.sigmoid(x)       
