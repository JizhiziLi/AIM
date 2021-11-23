import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from util import *
from network.resnet_mp import *

def conv_up_psp(in_channels, out_channels, up_sample):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=up_sample, mode='bilinear'))

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class AimNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34_mp()
        ##########################
        ### Encoder part - RESNET
        ##########################
        self.encoder0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            )
        self.mp0 = self.resnet.maxpool1
        self.encoder1 = nn.Sequential(
            self.resnet.layer1
            )
        self.mp1 = self.resnet.maxpool2
        self.encoder2 = self.resnet.layer2
        self.mp2 = self.resnet.maxpool3
        self.encoder3 = self.resnet.layer3
        self.mp3 = self.resnet.maxpool4
        self.encoder4 = self.resnet.layer4
        self.mp4 = self.resnet.maxpool5
        ##########################
        ### Decoder part - GLOBAL
        ##########################
        self.psp_module = PSPModule(512, 512, (1, 3, 5))
        self.psp4 = conv_up_psp(512, 256, 2)
        self.psp3 = conv_up_psp(512, 128, 4)
        self.psp2 = conv_up_psp(512, 64, 8)
        self.psp1 = conv_up_psp(512, 64, 16)
        self.decoder4_g = nn.Sequential(
            nn.Conv2d(1024,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear') )
        self.decoder4_g_se = SELayer(256)
        self.decoder3_g = nn.Sequential(
            nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear') )
        self.decoder3_g_se = SELayer(128)
        self.decoder2_g = nn.Sequential(
            nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.decoder2_g_se = SELayer(64)
        self.decoder1_g = nn.Sequential(
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.decoder1_g_se = SELayer(64)
        self.decoder0_g = nn.Sequential(
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.decoder0_g_spatial = nn.Conv2d(2,1,7,padding=3)
        self.decoder0_g_se = SELayer(64)
        self.decoder_final_g = nn.Conv2d(64,3,3,padding=1)
        ##########################
        ### Decoder part - LOCAL
        ##########################
        self.bridge_block = nn.Sequential(
            nn.Conv2d(512,512,3,dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.decoder4_l = nn.Sequential(
            nn.Conv2d(1024,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.decoder3_l = nn.Sequential(
            nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.decoder2_l = nn.Sequential(
            nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.decoder1_l = nn.Sequential(
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.decoder0_l = nn.Sequential(
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.decoder_final_l = nn.Conv2d(64,1,3,padding=1)

        
    def forward(self, input):

        #####################################
        ### Encoder part - MODIFIED RESNET
        #####################################
        e0 = self.encoder0(input)
        e0p, id0 = self.mp0(e0)
        e1p, id1 = self.mp1(e0p)
        e1 = self.encoder1(e1p)
        e2p, id2 = self.mp2(e1)
        e2 = self.encoder2(e2p)
        e3p, id3 = self.mp3(e2)
        e3 = self.encoder3(e3p)
        e4p, id4 = self.mp4(e3)
        e4 = self.encoder4(e4p)
        #####################################
        ### Decoder part - GLOBAL: Semantic
        #####################################
        psp = self.psp_module(e4)
        d4_g = self.decoder4_g(torch.cat((psp,e4),1))
        d4_g = self.decoder4_g_se(d4_g)
        d3_g = self.decoder3_g(torch.cat((self.psp4(psp),d4_g),1))
        d3_g = self.decoder3_g_se(d3_g)
        d2_g = self.decoder2_g(torch.cat((self.psp3(psp),d3_g),1))
        d2_g = self.decoder2_g_se(d2_g)
        d1_g = self.decoder1_g(torch.cat((self.psp2(psp),d2_g),1))
        d1_g = self.decoder1_g_se(d1_g)
        d0_g = self.decoder0_g(torch.cat((self.psp1(psp),d1_g),1))
        d0_g_avg = torch.mean(d0_g, dim=1,keepdim=True)
        d0_g_max, _ = torch.max(d0_g, dim=1,keepdim=True)
        d0_g_cat = torch.cat([d0_g_avg, d0_g_max], dim=1)
        d0_g_spatial = self.decoder0_g_spatial(d0_g_cat)
        d0_g_spatial_sigmoid = F.sigmoid(d0_g_spatial)
        d0_g = self.decoder0_g_se(d0_g)
        d0_g = self.decoder_final_g(d0_g)
        global_sigmoid = F.sigmoid(d0_g)
        #####################################
        ### Decoder part - LOCAL: Matting
        #####################################
        bb = self.bridge_block(e4)
        d4_l = self.decoder4_l(torch.cat((bb, e4),1))
        d3_l = F.max_unpool2d(d4_l, id4, kernel_size=2, stride=2)
        d3_l = self.decoder3_l(torch.cat((d3_l, e3),1))
        d2_l = F.max_unpool2d(d3_l, id3, kernel_size=2, stride=2)
        d2_l = self.decoder2_l(torch.cat((d2_l, e2),1))
        d1_l  = F.max_unpool2d(d2_l, id2, kernel_size=2, stride=2)
        d1_l = self.decoder1_l(torch.cat((d1_l, e1),1))
        d0_l  = F.max_unpool2d(d1_l, id1, kernel_size=2, stride=2)
        d0_l  = F.max_unpool2d(d0_l, id0, kernel_size=2, stride=2)
        d0_l = self.decoder0_l(torch.cat((d0_l, e0),1))
        d0_l = d0_l+d0_l*d0_g_spatial_sigmoid
        d0_l = self.decoder_final_l(d0_l)
        local_sigmoid = F.sigmoid(d0_l)
        ##########################
        ### Fusion net - G/L
        ##########################
        fusion_sigmoid = get_masked_local_from_global(global_sigmoid, local_sigmoid)
        return global_sigmoid, local_sigmoid, fusion_sigmoid