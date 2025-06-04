import torch
import torch.nn as nn
import torch.nn.functional as F
from res2net import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s
import torchvision
from CoordAttention import CoordAtt
### Feature-Alignment Pyramid Addition Network for Multimodal Change Detection (FPANet)
class FAModule(nn.Module):
    # Feature-Alignment Module (FAModule)
    def __init__(self,in_channel, out_chaanel):
        super(FAModule,self).__init__()
        self.en_conv_x1 = nn.Sequential(nn.Conv2d(in_channel, out_chaanel, kernel_size=(3,3), padding=1), nn.BatchNorm2d(out_chaanel), nn.ReLU(inplace=True))
        self.en_conv_x2 = nn.Sequential(nn.Conv2d(in_channel, out_chaanel, kernel_size=(3,3), padding=1), nn.BatchNorm2d(out_chaanel), nn.ReLU(inplace=True))
        self.mmd_x1 = nn.Sequential(nn.Conv2d(out_chaanel, out_chaanel, kernel_size=(3,3), padding=1), nn.BatchNorm2d(out_chaanel), nn.ReLU(inplace=True))
        self.mmd_x2 = nn.Sequential(nn.Conv2d(out_chaanel, out_chaanel, kernel_size=(3,3), padding=1), nn.BatchNorm2d(out_chaanel), nn.ReLU(inplace=True))
        self.de_conv_x1 = nn.Sequential(nn.Conv2d(out_chaanel, out_chaanel, kernel_size=(3,3), padding=1), nn.BatchNorm2d(out_chaanel), nn.ReLU(inplace=True))
        self.de_conv_x2 = nn.Sequential(nn.Conv2d(out_chaanel, out_chaanel, kernel_size=(3,3), padding=1), nn.BatchNorm2d(out_chaanel), nn.ReLU(inplace=True))
        self.CA = CoordAtt(inp=out_chaanel, oup=out_chaanel)

    def forward(self, x1, x2):
        x1 = self.en_conv_x1(x1)
        x1_mmd = self.mmd_x1(x1)
        x1 = self.de_conv_x1(x1_mmd)

        x2 = self.en_conv_x2(x2)
        x2_mmd = self.mmd_x2(x2)
        x2 = self.de_conv_x2(x2_mmd)
        return x1_mmd, x2_mmd, self.CA(abs(x1-x2))

class FPANet_NoSaim(nn.Module):
    # res2net based encoder decoder
    def __init__(self, pretrain=False):
        super(FPANet_NoSaim, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet1 = res2net50_v1b_26w_4s(pretrained=pretrain)
        self.resnet2 = res2net50_v1b_26w_4s(pretrained=pretrain)
        # self.resnet1 = res2net101_v1b_26w_4s(pretrained=pretrain)
        # self.resnet2 = res2net101_v1b_26w_4s(pretrained=pretrain)

        # self.CoordA_d5_d4 = CoordAtt(inp=64, oup=64)

        self.mmd_1 = FAModule(in_channel=64, out_chaanel=64) # output mmd_out, x
        self.mmd_2 = FAModule(in_channel=256, out_chaanel=64)
        self.mmd_3 = FAModule(in_channel=512, out_chaanel=64)
        self.mmd_4 = FAModule(in_channel=1024, out_chaanel=64)
        self.mmd_5 = FAModule(in_channel=2048, out_chaanel=64)

        self.d5_d4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.d4_d3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.d3_d2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.d2_d1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.d5_d4_d3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.d4_d3_d2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.d3_d2_d1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.d5_d4_d3_d2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.d4_d3_d2_d1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.d5_dem_4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.d5_d4_d3_d2_d1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # Decoder
        self.level3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), CoordAtt(inp=64, oup=64))
        self.level2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), CoordAtt(inp=64, oup=64))
        self.level1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), CoordAtt(inp=64, oup=64))
        self.d5_dem_5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), CoordAtt(inp=64, oup=64))

        # self.level3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # self.level2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # self.level1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        #
        # self.d5_dem_5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))


        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.out4 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=(3,3), padding=1))
        self.out3 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=(3,3), padding=1))
        self.out2 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=(3,3), padding=1))
        self.out1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=(3,3), padding=1))

    def forward(self, i1, i2):
        mmds = []
        preds = []
        # x = torch.cat((x1, x2), 1)
        inSize = i1.size()[2:]

        # ''' Siamese-Encoder I1-image
        i1 = self.resnet1.conv1(i1)
        i1 = self.resnet1.bn1(i1)
        i1 = self.resnet1.relu(i1)

        i1_x1 = self.resnet1.maxpool(i1)      # bs, 64, 64, 64

        # ---- low-level features ----
        i1_x2 = self.resnet1.layer1(i1_x1)      # bs, 256, 64, 64

        i1_x3 = self.resnet1.layer2(i1_x2)     # bs, 512, 32, 32

        i1_x4 = self.resnet1.layer3(i1_x3)     # bs, 1024, 16, 16

        i1_x5 = self.resnet1.layer4(i1_x4)     # bs, 2048, 8, 8
        # '''

        # ''' Siamese-Encoder I2-image
        i2 = self.resnet2.conv1(i2)
        i2 = self.resnet2.bn1(i2)
        i2 = self.resnet2.relu(i2)

        i2_x1 = self.resnet2.maxpool(i2)  # bs, 64, 64*64

        # ---- low-level features ----
        i2_x2 = self.resnet2.layer1(i2_x1)  # bs, 256, 64, 64


        i2_x3 = self.resnet2.layer2(i2_x2)  # bs, 512, 32, 32

        i2_x4 = self.resnet2.layer3(i2_x3)  # bs, 1024, 16, 16

        i2_x5 = self.resnet2.layer4(i2_x4)  # bs, 2048, 8, 8

        # '''

        mmd_i1_x1, mmd_i2_x1, diff_1  = self.mmd_1(i1_x1, i2_x1)
        mmd_i1_x2, mmd_i2_x2, diff_2 = self.mmd_2(i1_x2, i2_x2)
        mmd_i1_x3, mmd_i2_x3, diff_3 = self.mmd_3(i1_x3, i2_x3)
        mmd_i1_x4, mmd_i2_x4, diff_4 = self.mmd_4(i1_x4, i2_x4)
        mmd_i1_x5, mmd_i2_x5, diff_5 = self.mmd_5(i1_x5, i2_x5)
        mmds.append((mmd_i1_x1, mmd_i2_x1))
        mmds.append((mmd_i1_x2, mmd_i2_x2))
        mmds.append((mmd_i1_x3, mmd_i2_x3))
        mmds.append((mmd_i1_x4, mmd_i2_x4))
        mmds.append((mmd_i1_x5, mmd_i2_x5))

        d5_4 = self.d5_d4(F.upsample(diff_5, size=diff_4.size()[2:], mode='bilinear') + diff_4)
        d4_3 = self.d4_d3(F.upsample(diff_4, size=diff_3.size()[2:], mode='bilinear') + diff_3)
        d3_2 = self.d3_d2(F.upsample(diff_3, size=diff_2.size()[2:], mode='bilinear') + diff_2)
        d2_1 = self.d2_d1(F.upsample(diff_2, size=diff_1.size()[2:], mode='bilinear') + diff_1)


        d5_4_3 = self.d5_d4_d3(F.upsample(d5_4, size=d4_3.size()[2:], mode='bilinear') + d4_3)
        d4_3_2 = self.d4_d3_d2(F.upsample(d4_3, size=d3_2.size()[2:], mode='bilinear') + d3_2)
        d3_2_1 = self.d3_d2_d1(F.upsample(d3_2, size=d2_1.size()[2:], mode='bilinear') + d2_1)


        d5_4_3_2 = self.d5_d4_d3_d2(F.upsample(d5_4_3, size=d4_3_2.size()[2:], mode='bilinear') + d4_3_2)
        d4_3_2_1 = self.d4_d3_d2_d1(F.upsample(d4_3_2, size=d3_2_1.size()[2:], mode='bilinear') + d3_2_1)

        d5_dem_4 = self.d5_dem_4(d5_4_3_2)
        d5_4_3_2_1 = self.d5_d4_d3_d2_d1(F.upsample(d5_dem_4, size=d4_3_2_1.size()[2:], mode='bilinear') + d4_3_2_1)

        level4 = d5_4
        # level4 = self.CoordA_d5_d4(d5_4)
        level3 = self.level3(d4_3 + d5_4_3)
        level2 = self.level2(d3_2 + d4_3_2 + d5_4_3_2)
        level1 = self.level1(d2_1 + d3_2_1 + d4_3_2_1 + d5_4_3_2_1)

        d5_dem_5 = self.d5_dem_5(diff_5)

        output4 = self.output4(F.upsample(d5_dem_5, size=level4.size()[2:], mode='bilinear') + level4)
        output3 = self.output3(F.upsample(output4, size=level3.size()[2:], mode='bilinear') + level3)
        output2 = self.output2(F.upsample(output3, size=level2.size()[2:], mode='bilinear') + level2)
        output1 = self.output1(F.upsample(output2, size=level1.size()[2:], mode='bilinear') + level1)

        out4 = self.out4(output4)
        out3 = self.out3(output3)
        out2 = self.out2(output2)
        out1 = self.out1(output1)

        preds.append(out1)
        preds.append(out2)
        preds.append(out3)
        preds.append(out4)

        pred = F.upsample(out1, size=inSize, mode='bilinear')

        return pred, preds, mmds

class PAM(nn.Module):
    """
    This code refers to "Dual attention network for scene segmentation"Position attention module".
    Ref from SAGAN
    """
    def __init__(self, in_dim):
        super(PAM, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """

        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

if __name__ == '__main__':
    net = FPANet_NoSaim(pretrain=False).cuda(3)
    img_1 = torch.randn(2, 3, 256, 256).cuda(3)
    img_2 = torch.randn(2, 3, 256, 256).cuda(3)
    pred, preds, mmds = net(img_1, img_2)
    print(pred)
    print(preds.shape())
    print(mmds.shape())