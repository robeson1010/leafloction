from torchvision import models
import torchvision
import torch.nn as nn
import torch

from Decoder import conv3x3,ConvRelu,DecoderBlock,DecoderBlockV2
class UnetResnet34(nn.Module):
    def __init__(self,
                 num_classes=2,
                 num_filters=32,
                 pretrained=True,
                 is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)        
        
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4
        self.avgpool = self.encoder.avgpool
        self.fc=nn.Linear(in_features=512, out_features=data.c, bias=True)

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
#         self.trans = conv_layer(2,1,1,leaky=0.1)
        self.trans2 = conv_layer(2,3,1,leaky=0.1)
#         self.ReLU=nn.ReLU(inplace=True)

                
    def forward(self, x):
        conv1 = self.conv1(x)
#         print(conv1.shape)
        conv2 = self.conv2(conv1)
#         print(conv2.shape)
        conv3 = self.conv3(conv2)
#         print(conv3.shape)
        conv4 = self.conv4(conv3)
#         print(conv4.shape)
        conv5 = self.conv5(conv4)
#         print(conv5.shape)
#         print(conv5.shape)
        out1=self.avgpool(conv5)
        out1 = out1.reshape(out1.size(0), -1)
        out1=self.fc(out1)
        center = self.center(self.pool(conv5))
#         print(center.shape)
        dec5 = self.dec5(torch.cat([center, conv5], 1))
#         print(dec5.shape)
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
#         print(dec4.shape)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
#         print(dec3.shape)
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
#         print(dec2.shape)
        dec1 = self.dec1(dec2)
#         print(dec1.shape)
        dec0 = self.dec0(dec1)
#         print(dec0.shape)
        x_out = self.final(dec0)
#         newx=self.trans(x_out)
# #         print(newx.shape,x.shape)
# #         x=x+newx[:,1:2,:]*x
#         x=x+newx*x
        newx=self.trans2(x_out)
#         x_out = torch.sigmoid(self.final(dec0))
#         x=x_out.expand_as(x)
#         x=x*x_out+x
        conv6 = self.conv1(newx)
        conv6 = self.conv2(conv6)
        conv6 = self.conv3(conv6)
        conv6 = self.conv4(conv6)
        conv6 = self.conv5(conv6)
        out2=self.avgpool(conv6)
        out2 = out2.reshape(out2.size(0), -1)
        out2=self.fc(out2)
        return out1,out2