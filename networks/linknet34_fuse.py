import torch
from torchvision import models
from .basic_blocks import *
import math
import torch.nn.functional as F

class LinkNet34_fuse(nn.Module):
    def __init__(self, block_size='1,2,4'):
        super(LinkNet34_fuse, self).__init__()
        filters = [64, 128, 256, 512]
        self.block_size = [int(s) for s in block_size.split(',')]

        # img
        resnet = models.resnet34(pretrained=True)
        self.firstconv1 = nn.Conv2d(1, filters[0], kernel_size=7, stride=2, padding=3)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4


        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], filters[0] // 2, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(filters[0] // 2, filters[0], 3, padding=1)
        self.finalrelu2 = nonlinearity

        ## addinfo, e.g, gps_map, lidar_map
        resnet1 = models.resnet34(pretrained=True)
        self.firstconv1_add = nn.Conv2d(1, filters[0], kernel_size=7, stride=2, padding=3)
        self.firstbn_add = resnet1.bn1
        self.firstrelu_add = resnet1.relu
        self.firstmaxpool_add = resnet1.maxpool

        self.encoder1_add = resnet1.layer1
        self.encoder2_add = resnet1.layer2
        self.encoder3_add = resnet1.layer3
        self.encoder4_add = resnet1.layer4

        self.fuse1=Cross_Mod(filters[0],filters[0])
        self.fuse2 = Cross_Mod(filters[1], filters[1])
        self.fuse3 = Cross_Mod(filters[2], filters[2])
        self.fuse4 = Cross_Mod(filters[3], filters[3])
        self.conv1= nn.Conv2d(2*filters[0], filters[0], 3, padding=1)
        self.conv2 = nn.Conv2d(2 * filters[1], filters[1], 3, padding=1)
        self.conv3 = nn.Conv2d(2 * filters[2], filters[2], 3, padding=1)
        self.conv4 = nn.Conv2d(2 * filters[3], filters[3], 3, padding=1)



        self.finalconv = nn.Conv2d(filters[0], 1, 3, padding=1)

    def forward(self, inputs):
        x = inputs[:, 0, :, :]  # image
        x = x.unsqueeze(1)
        add = inputs[:, 1:, :, :]  # gps_map or lidar_map
        # 进入编码-解码结构前有个将原图像做卷积步骤
        x = self.firstconv1(x)
        add = self.firstconv1_add(add)
        x = self.firstmaxpool(self.firstrelu(self.firstbn(x)))
        add = self.firstmaxpool_add(self.firstrelu_add(self.firstbn_add(add)))
        # 每一层的图像和adding的额外信息例如gps都输入DEM模块，输出增强的图像和adding特征信息，然后再输入下一层以此循环
        x_e1 = self.encoder1(x)
        add_e1 = self.encoder1_add(add)
        x_e1,add_e1= self.fuse1(x_e1,add_e1)

        x_fuse1= self.conv1(torch.cat([x_e1,add_e1],1))

        x_e2 = self.encoder2(x_e1)
        add_e2 = self.encoder2_add(add_e1)
        x_e2, add_e2 = self.fuse2(x_e2, add_e2)
        x_fuse2 = self.conv2(torch.cat([x_e2, add_e2], 1))

        x_e3 = self.encoder3(x_e2)
        add_e3 = self.encoder3_add(add_e2)
        x_e3, add_e3 = self.fuse3(x_e3, add_e3)
        x_fuse3 = self.conv3(torch.cat([x_e3, add_e3], 1))

        x_e4 = self.encoder4(x_e3)
        add_e4 = self.encoder4_add(add_e3)

        x_e4, add_e4 = self.fuse4(x_e4, add_e4)
        x_fuse4 = self.conv4(torch.cat([x_e4, add_e4], 1))
       # 传递增强信息时还有跳跃连接
        # Decoder
        x_d4 = self.decoder4(x_fuse4) + x_fuse3


        x_d3 = self.decoder3(x_d4) + x_fuse2


        x_d2 = self.decoder2(x_d3) + x_fuse1


        x_d1 = self.decoder1(x_d2)


        x_out = self.finalrelu1(self.finaldeconv1(x_d1))
        x_out = self.finalrelu2(self.finalconv2(x_out))

        out = self.finalconv(x_out)  # b*1*h*w
        return torch.sigmoid(out)


