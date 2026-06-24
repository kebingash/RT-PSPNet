import math

import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

model_urls = {
    'resnet50': 'https://github.com/bubbliiiing/pspnet-pytorch/releases/download/v1.0/resnet50s-a75c83cf.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                               bias=False, groups=planes)
        self.bn2 = norm_layer(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, dilated=True, deep_base=True, norm_layer=nn.BatchNorm2d):
        self.inplanes = 64 if deep_base else 64
        super(ResNet, self).__init__()
        if deep_base:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 96, layers[0], norm_layer=norm_layer)
        # self.layer2     = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        # if dilated:
        #     self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
        #                                    dilation=2, norm_layer=norm_layer)
        #     self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
        #                                     dilation=4, norm_layer=norm_layer)
        if dilated:
            # self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            #                                dilation=2, norm_layer=norm_layer)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            # self.layer3 = self._make_layer(block, 128, layers[1], stride=1,
            #                                dilation=2, norm_layer=norm_layer)
        else:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=1,
                                           norm_layer=norm_layer)
            # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
            #                                norm_layer=norm_layer)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, multi_grid=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        multi_dilations = [4, 8, 16]
        if multi_grid:
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilations[0],
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if multi_grid:
                layers.append(block(self.inplanes, planes, dilation=multi_dilations[i],
                                    previous_dilation=dilation, norm_layer=norm_layer))
            else:
                layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        #
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        #
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        x = self.relu(self.bn1(self.conv1(x)))
        # x = self.relu2(self.bn2(self.conv2(x)))
        # x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        feat1 = x

        x = self.layer1(x)
        feat2 = x
        x = self.maxpool(x)
        x = self.layer2(x)
        feat3 = x

        # return x
        return feat1, feat2, feat3


def resnet50(pretrained=False, **kwargs):
    # model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model = ResNet(Bottleneck, [3, 4], **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['resnet50'], "./model_data"), strict=False)
    return model

###Shallow lightweight resnet###
class Resnet(nn.Module):
    def __init__(self, dilate_scale=8, pretrained=True):
        super(Resnet, self).__init__()
        from functools import partial
        model = resnet50(pretrained)

        # --------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,1024和30,30,2048
        # --------------------------------------------------------------------------------------------#
        # if dilate_scale == 8:
        #     model.layer3.apply(partial(self._nostride_dilate, dilate=2))
        #     model.layer4.apply(partial(self._nostride_dilate, dilate=4))
        # elif dilate_scale == 16:
        #     model.layer4.apply(partial(self._nostride_dilate, dilate=2))

        self.conv1 = model.conv1[0]
        self.bn1 = model.conv1[1]
        self.relu1 = model.conv1[2]
        self.conv2 = model.conv1[3]
        self.bn2 = model.conv1[4]
        self.relu2 = model.conv1[5]
        self.conv3 = model.conv1[6]
        self.bn3 = model.bn1
        self.relu3 = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        # self.layer3 = model.layer3
        # self.layer4 = model.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        feat1 = x

        x = self.layer1(x)
        feat2 = x
        x = self.maxpool(x)
        x = self.layer2(x)
        feat3 = x
        # x_aux = self.layer3(x)
        # feat4 = x_aux
        # x = self.layer4(x_aux)
        # feat5 = x
        return feat1, feat2, feat3
        # return x


# class MobileNetV2(nn.Module):
#     def __init__(self, downsample_factor=8, pretrained=True):
#         super(MobileNetV2, self).__init__()
#         from functools import partial
#
#         model = mobilenetv2(pretrained)
#         self.features = model.features[:-1]
#
#         self.total_idx = len(self.features)
#         self.down_idx = [2, 4, 7, 14]
#
#         # --------------------------------------------------------------------------------------------#
#         #   根据下采样因子修改卷积的步长与膨胀系数
#         #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,320和30,30,96
#         # --------------------------------------------------------------------------------------------#
#         if downsample_factor == 8:
#             for i in range(self.down_idx[-2], self.down_idx[-1]):
#                 self.features[i].apply(partial(self._nostride_dilate, dilate=2))
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(partial(self._nostride_dilate, dilate=4))
#         elif downsample_factor == 16:
#             for i in range(self.down_idx[-1], self.total_idx):
#                 self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    # def _nostride_dilate(self, m, dilate):
    #     classname = m.__class__.__name__
    #     if classname.find('Conv') != -1:
    #         if m.stride == (2, 2):
    #             m.stride = (1, 1)
    #             if m.kernel_size == (3, 3):
    #                 m.dilation = (dilate // 2, dilate // 2)
    #                 m.padding = (dilate // 2, dilate // 2)
    #         else:
    #             if m.kernel_size == (3, 3):
    #                 m.dilation = (dilate, dilate)
    #                 m.padding = (dilate, dilate)
    #
    # def forward(self, x):
    #     # x_aux = self.features[:14](x)
    #     # x = self.features[14:](x_aux)
    #
    #     # return x_aux, x
    #     feat1 = self.features[:2](x)
    #     feat2 = self.features[2:4](feat1)
    #     feat3 = self.features[4:7](feat2)
    #     feat4 = self.features[7:11](feat3)
    #     x_aux = self.features[11:14](feat4)
    #     feat5 = self.features[14:17](x_aux)
    #     feat6 = self.features[17:](feat5)
    #
    #     return x_aux, feat1, feat2, feat3, feat4, feat5, feat6

###CPPM###
class _PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        # out_channels = in_channels // (len(pool_sizes)+1)
        # out_channels = in_channels // (len(pool_sizes))
        out_channels = in_channels
        # print(in_channels, out_channels, len(pool_sizes), in_channels // (len(pool_sizes))*2)
        # out_channels = in_channels
        # -----------------------------------------------------#
        #   分区域进行平均池化
        #   30, 30, 320 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 = 30, 30, 640
        # -----------------------------------------------------#
        self.stages = nn.ModuleList(
            [self._make_stages(in_channels, out_channels, pool_size, norm_layer) for pool_size in pool_sizes])

        # 30, 30, 640 -> 30, 30, 80
        self.bottleneck = nn.Sequential(
            # nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                      bias=False, groups=out_channels),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend(
            [F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        # output = self.bottleneck(torch.cat(pyramids, dim=1))
        output = self.bottleneck(pyramids[0] + pyramids[1] + pyramids[2])

        return output


class PSPNet(nn.Module):
    def __init__(self, num_classes, downsample_factor, backbone="shallow_lightweight_resnet", pretrained=True, aux_branch=False):
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        if backbone == "shallow_lightweight_resnet":
            self.backbone = Resnet(downsample_factor, pretrained)
            aux_channel = 1024
            # aux_channel = 512
            out_channel = 512
            # out_channel = 512
        # elif backbone == "mobilenet":
        #     # ----------------------------------#
        #     #   获得两个特征层
        #     #   f4为辅助分支    [30,30,96]
        #     #   o为主干部分     [30,30,320]
        #     # ----------------------------------#
        #     self.backbone = MobileNetV2(downsample_factor, pretrained)
        #     aux_channel = 96
        #     out_channel = 320
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

        # --------------------------------------------------------------#
        #	PSP模块，分区域进行池化
        #   分别分割成1x1的区域，2x2的区域，3x3的区域，6x6的区域
        #   30,30,320 -> 30,30,80 -> 30,30,21
        # --------------------------------------------------------------#
        self.master_branch = nn.Sequential(
            # _PSPModule(out_channel, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            _PSPModule(out_channel, pool_sizes=[16, 32], norm_layer=norm_layer)
            # _PSPModule(out_channel//2, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            # nn.Conv2d(out_channel//4, num_classes, kernel_size=1)
        )
        # self.master_branch_ = nn.Sequential(
        #     # _PSPModule(out_channel, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer)
        #     _PSPModule(out_channel, pool_sizes=[8, 16], norm_layer=norm_layer)
        #     # _PSPModule(out_channel//2, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
        #     # nn.Conv2d(out_channel//4, num_classes, kernel_size=1)
        # )

        self.aux_branch = aux_branch

        if self.aux_branch:
            # ---------------------------------------------------#
            #	利用特征获得预测结果
            #   30, 30, 96 -> 30, 30, 40 -> 30, 30, 21
            # ---------------------------------------------------#
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(aux_channel, out_channel // 8, kernel_size=3, padding=1, bias=False),
                norm_layer(out_channel // 8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_channel // 8, num_classes, kernel_size=1)
            )

        self.out_channel = out_channel
        # self.initialize_weights(self.master_branch)
        self.upsamplex2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsamplex4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.maxpool = nn.MaxPool2d(2, 2)
        # self.conv1x1_1 = nn.Conv2d(160,64,1)

        # self.conv1x1_2_ = nn.Conv2d(32, 64, 1)
        # self.conv1x1_2 = nn.Conv2d(24, 64, 1)
        # self.conv1x1_3 = nn.Conv2d(16, 64, 1)
        # self.conv3x3 = nn.Conv2d(64, 2, 3, padding=1)

        self.conv1x1_1 = nn.Conv2d(384, 2, 1)
        self.conv1x1_2 = nn.Conv2d(512, 64, 1)
        self.conv1x1_3 = nn.Conv2d(384, 64, 1)
        self.conv1x1_4 = nn.Conv2d(512, 128, 1)
        self.conv3x3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3x3_1 = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.conv3x3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.finalconv = nn.Conv2d(64, 2, 1)
        self.class_conv = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        # input_size = (x.size()[2], x.size()[3])
        # x = self.backbone(x)
        # output = self.conv1x1_2(self.master_branch(x))
        # output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        # if self.aux_branch:
        #     output_aux = self.auxiliary_branch(x_aux)
        #     output_aux = F.interpolate(output_aux, size=input_size, mode='bilinear', align_corners=True)
        #     return output_aux, output
        # else:
        #     return output

        # input_size = (x.size()[2], x.size()[3])
        # norm_layer = nn.BatchNorm2d
        feats = self.backbone(x)
        # # input_size = [feats[-1].size()[2], feats[-1].size()[3]]
        # output = self.conv1x1_4(feats[-1]) + self.master_branch(feats[-1])
        ###Feature Fusion###
        output = feats[-1] + self.master_branch(feats[-1])
        # output = self.master_branch(feats[-1])
        # # output = feats[-1]
        #
        output = self.conv1x1_2(self.upsamplex2(output)) + self.conv1x1_3(feats[1])
        # output = self.conv1x1_2(output)
        #
        # # output = self.riam.forward(output)
        output = self.finalconv(self.conv3x3(output))

        # return output, self.upsamplex2(self.conv1x1_4(feats[-1])), self.conv1x1_1(feats[1])
        return output

    def initialize_weights(self, *models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(1e-4)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0001)
                    m.bias.data.zero_()
