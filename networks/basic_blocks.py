import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
nonlinearity = partial(F.relu, inplace=True)
import matplotlib.pyplot as plt
# class Exchange(nn.Module):
#     def __init__(self):
#         super(Exchange, self).__init__()
#
#     def forward(self, x, bn, bn_threshold):
#         bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
#         x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
#         x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
#         x1[:, bn1 < bn_threshold] = x[1][:, bn1 < bn_threshold]
#         x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
#         x2[:, bn2 < bn_threshold] = x[0][:, bn2 < bn_threshold]
#         return [x1, x2]
class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, bn, bn_threshold):
        bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
        #灏辨槸澶т簬闃堝€肩殑閭ｄ簺閫氶亾淇濈暀锛屽皬浜庨槇鍊肩殑閭ｄ簺閫氶亾鍙栧彟澶栦竴涓殑鍊�
        # print(bn1)
        # bn_threshold1 = search_threshold(bn1,"grad")
        # bn_threshold2 = search_threshold(bn2, "grad")
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
        x1[:, bn1 < bn_threshold] = x[1][:, bn1 < bn_threshold]
        x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
        x2[:, bn2 < bn_threshold] = x[0][:, bn2 < bn_threshold]
        # x[0][:, bn1 < bn_threshold] = x[1][:, bn1 < bn_threshold]
        # x[1][:, bn2 < bn_threshold] = x[0][:, bn2 < bn_threshold]
        # print('bn',bn1 < bn_threshold)
        xa= torch.mean(x[0][:, bn1 < bn_threshold],dim=1)
        xb=  torch.mean(x[1][:, bn2 < bn_threshold],dim=1)
        xc = torch.mean(x[0][:, bn1 >= bn_threshold], dim=1)
        xd = torch.mean(x[1][:, bn2 >= bn_threshold], dim=1)
        plt.subplot(2,2,1)
        plt.imshow(xa[0].detach().cpu())
        plt.colorbar(label='color bar settings')
        #plt.clim(-0.1, 0.04)
        plt.subplot(2, 2, 2)
        plt.imshow(xb[0].detach().cpu())
        #plt.clim(-0.1, 0.04)
        plt.colorbar(label='color bar settings')
        plt.subplot(2, 2, 3)
        plt.imshow(xc[0].detach().cpu())
        #plt.clim(-0.3, 0.5)
        plt.colorbar(label='color bar settings')
        plt.subplot(2, 2, 4)
        #plt.clim(-0.3, 0.5)
        plt.imshow(xd[0].detach().cpu())
        plt.colorbar(label='color bar settings')
        plt.show()
        return x1,x2

class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel=2):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(int(num_parallel)):

            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]
def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                    stride=stride, padding=1, bias=bias))


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                    stride=stride, padding=0, bias=bias))

class DBlock(nn.Module):
    def __init__(self, channel):
        super(DBlock, self).__init__()
        self.dilate1 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=8, padding=8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out
class DBlock_parallel(nn.Module):
    def __init__(self, channel,num_parallel):
        super(DBlock_parallel, self).__init__()
        self.dilate1 = ModuleParallel(nn.Conv2d(
            channel, channel, kernel_size=3, dilation=1, padding=1))
        self.dilate2 = ModuleParallel(nn.Conv2d(
            channel, channel, kernel_size=3, dilation=2, padding=2))
        self.dilate3 = ModuleParallel(nn.Conv2d(
            channel, channel, kernel_size=3, dilation=4, padding=4))
        self.dilate4 = ModuleParallel(nn.Conv2d(
            channel, channel, kernel_size=3, dilation=8, padding=8))
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
        self.num_parallel=num_parallel
    def forward(self, x):
        dilate1_out = self.relu(self.dilate1(x))
        dilate2_out = self.relu(self.dilate2(dilate1_out))
        dilate3_out = self.relu(self.dilate3(dilate2_out))
        dilate4_out = self.relu(self.dilate4(dilate3_out))
        out = [x[l] + dilate1_out[l] + dilate2_out[l] + dilate3_out[l] + dilate4_out[l] for l in range(self.num_parallel)]

        return out
class DecoderBlock_parallel_exchange(nn.Module):
    def __init__(self, in_channels, n_filters,num_parallel,bn_threshold):
        super(DecoderBlock_parallel_exchange, self).__init__()

        self.conv1 = conv1x1(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm2dParallel(in_channels // 4, num_parallel)
        self.relu1 =  ModuleParallel(nn.ReLU(inplace=True))
        self.deconv2 = ModuleParallel(nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
        ))
        self.bn2 = BatchNorm2dParallel(in_channels // 4, num_parallel)
        self.relu2 = ModuleParallel(nn.ReLU(inplace=True))
        self.conv3 = conv1x1(in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm2dParallel(n_filters, num_parallel)
        self.relu3 = ModuleParallel(nn.ReLU(inplace=True))
        self.exchange = Exchange()
        self.bn_threshold = bn_threshold
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        if len(x) > 1:
            x = self.exchange(x, self.bn2_list, self.bn_threshold)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x = self.relu3(x)
        return x

class DecoderBlock_parallel(nn.Module):
    def __init__(self, in_channels, n_filters,num_parallel):
        super(DecoderBlock_parallel, self).__init__()

        self.conv1 = conv1x1(in_channels, in_channels // 4, 1)
        self.norm1 = BatchNorm2dParallel(in_channels // 4, num_parallel)
        self.relu1 =  ModuleParallel(nn.ReLU(inplace=True))
        self.deconv2 = ModuleParallel(nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
        ))
        self.norm2 = BatchNorm2dParallel(in_channels // 4, num_parallel)
        self.relu2 = ModuleParallel(nn.ReLU(inplace=True))
        self.conv3 = conv1x1(in_channels // 4, n_filters, 1)
        self.norm3 = BatchNorm2dParallel(n_filters, num_parallel)
        self.relu3 = ModuleParallel(nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class Cross_Mod(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ = in_ch
        self.out_ = out_ch
        self.sconv = nn.Conv2d(in_ch, out_ch, 3, 1,1)  # 进行特征图尺寸减半(gail )
        self.conv_cat = nn.Conv2d(2 * out_ch, 2 * out_ch, 3, 1, 1)  # 用于拼接后的卷积核
        self.convs = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, 1, 1)
                                   , nn.Sigmoid())

    def forward(self, ir, vi):
        ir = self.sconv(ir)
        vi = self.sconv(vi)
        img_cat = torch.cat((vi, ir), dim=1)
        img_conv = self.conv_cat(img_cat)

        # 平均拆分拼接后的特征
        split_ir = img_conv[:, self.out_:]  # 前一半特征
        split_vi = img_conv[:, :self.out_]  # 后一半特征

        # IR图像处理流程
        ir_1 = self.convs(split_ir)
        ir_2 = self.convs(split_ir)
        ir_mul = torch.mul(ir, ir_1)
        ir_add = ir_1 + ir_2

        # VI图像处理流程
        vi_1 = self.convs(split_vi)
        vi_2 = self.convs(split_vi)
        vi_mul = torch.mul(vi, vi_1)
        vi_add = vi_1 + vi_2

        return ir_add, vi_add