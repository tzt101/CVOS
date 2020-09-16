"""
Copyright (C) University of Science and Technology of China.
Licensed under the MIT License.
"""

from network import DeepLab
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import gc
from torchvision import models
from torch.nn.functional import grid_sample
import math
import numpy as np
import torch.utils.model_zoo as model_zoo

def load_weights(cnn_model, weights):
    """
    argus:
    :param cnn_model: the cnn networks need to load weights
    :param weights: the pretrained weigths
    :return: no return
    """
    pre_dict = cnn_model.state_dict()
    for key, val in weights.items():
        if key[0:7] == 'module.': # the pretrained networks was trained on multi-GPU
            key = key[7:] # remove 'module.' from the key
        if key in pre_dict.keys():
            if isinstance(val, Parameter):
                val = val.data
            pre_dict[key].copy_(val)
    cnn_model.load_state_dict(pre_dict)

def l2_norm(x):
    norm = torch.norm(x, 2, 1)
    norm = torch.unsqueeze(norm, dim=1).float()
    x = x / norm.clamp(min=1e-8)
    return x

def CosineSimilarityCalculate2(f, mf, mb, k=20):
    # norm
    f = l2_norm(f)
    mf = l2_norm(mf)
    mb = l2_norm(mb)
    #
    nf, cf, hf, wf = f.shape
    f = f.permute((0,2,3,1)).contiguous()
    mf = mf.permute((0, 2, 3, 1)).contiguous()
    mb = mb.permute((0, 2, 3, 1)).contiguous()
    # reshape
    f = f.view(nf*hf*wf, -1)
    mf = mf.view(nf * hf * wf, -1)
    mb = mb.view(nf * hf * wf, -1)
    #
    outf = torch.mm(f, mf.permute((1, 0)))
    outb = torch.mm(f, mb.permute((1, 0)))
    # top k
    outf, _ = outf.sort(1, descending=True)
    outf = outf[:,0:k].mean(dim=1)
    outb, _ = outb.sort(1, descending=True)
    outb = outb[:,0:k].mean(dim=1)
    # reshape
    outf = outf.view(nf, 1, hf, wf)
    outb = outb.view(nf, 1, hf, wf)
    # upsample
    outf = F.upsample(outf, scale_factor=8, mode='bilinear', align_corners=True)
    outb = F.upsample(outb, scale_factor=8, mode='bilinear', align_corners=True)
    #
    out = torch.cat([outb, outf], 1)
    return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        deeplab = DeepLab.Res_Deeplab()

        self.conv1 = deeplab.conv1
        self.bn1 = deeplab.bn1
        self.relu = deeplab.relu  # 1/2, 64
        self.maxpool = deeplab.maxpool

        self.res2 = deeplab.layer1  # 1/4, 256
        self.res3 = deeplab.layer2  # 1/8, 512
        self.res4 = deeplab.layer3  # 1/8, 1024
        self.res5 = deeplab.layer4  # 1/8, 2048
        self.pred = nn.Conv2d(2048, 256, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        # freeze BNs and Cnovs
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                # m.eval()
                for p in m.parameters():
                    p.requires_grad = False

            # if isinstance(m, nn.Conv2d):
            #     for p in m.parameters():
            #         p.requires_grad = False

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # x = (x - Variable(self.mean)) / Variable(self.std)

        x = self.conv1(x)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 64
        r3 = self.res3(r2)  # 1/8, 128
        r4 = self.res4(r3)  # 1/8, 256
        r5 = self.res5(r4)  # 1/8, 512
        out = self.pred(r5)
        return out

class MotionEncoderPool(nn.Module):
    def __init__(self):
        super(MotionEncoderPool, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x1 = self.pool(x) # 1/2
        x1 = x1 / 2
        x2 = self.pool(x1) # 1/4
        x2 = x2 /2
        x3 = self.pool(x2) # 1/8
        x3 = x3 / 2
        return x3-1


class WarperConv(nn.Module):

    def feat_warp(self, f, m):
        # f is the feat and m is the motion
        # reshape
        m = m.permute([0, 2, 3, 1])
        warped_f = grid_sample(f, m, padding_mode='border')
        return warped_f

    def __init__(self):
        super(WarperConv, self).__init__()
        # the motion vector is [batch, 2, h, w] and be normalized

    def forward(self, f, m):
        new_f = self.feat_warp(f, m)
        return new_f

# ResNet

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    return model

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))
    return model

class AttentionEncoder(nn.Module):
    def __init__(self):
        super(AttentionEncoder, self).__init__()
        base = resnet18(pretrained=True)

        # self.Conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # n = self.Conv1.kernel_size[0] * self.Conv1.kernel_size[1] * self.Conv1.out_channels
        # self.Conv1.weight.data.normal_(0, math.sqrt(2. / n))
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.res2 = base.layer1 # 1/4 64
        self.res3 = base.layer2 # 1/8 128
        self.res4 = base.layer3 # 1/8 256
        self.res5 = base.layer4 # 1/8 512

        self.pred = nn.Conv2d(512, 2, 3, 1, 1, 1, 1, False)
        n = self.pred.kernel_size[0] * self.pred.kernel_size[1] * self.pred.out_channels
        self.pred.weight.data.normal_(0, math.sqrt(2. / n))

        # freeze BNs and Cnovs
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                # m.eval()
                for p in m.parameters():
                    p.requires_grad = False
            # if isinstance(m, nn.Conv2d):
            #     for p in m.parameters():
            #         p.requires_grad = False

    def forward(self, motion, gb):
        gb = gb.float()
        x = torch.cat([motion,gb], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        r2 = self.res2(x)  # 1/4, 64
        r3 = self.res3(r2)  # 1/8, 128
        r4 = self.res4(r3)  # 1/8, 256
        r5 = self.res5(r4) # 1/8, 512
        p = self.pred(r5) # 1/8, 2
        p_a = F.softmax(p, 1) # attention map
        p_l = F.upsample(p, scale_factor=8, mode='bilinear', align_corners=True)
        p_a = p_a[:,1,:,:]
        p_a = torch.unsqueeze(p_a, dim=1).float() # b, 1, h, w


        return p_a, p_l

class AttentionOperate(nn.Module):
    def __init__(self):
        super(AttentionOperate, self).__init__()
        self.f = nn.Conv2d(256, 256, 1, 1, 0, 1, 1, False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        # freeze BNs and Cnovs
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                # m.eval()
                for p in m.parameters():
                    p.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, feat, map):
        x = self.bn(feat)
        x = self.relu(x)
        x = self.f(x)
        out = feat + x * map
        return out



class ResidualEncoder(nn.Module):
    def __init__(self):
        super(ResidualEncoder, self).__init__()
        resnet = resnet50(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024
        self.res5 = resnet.layer4  # 1/8, 2048
        self.pred = nn.Conv2d(2048, 256, kernel_size=1)


        # freeze BNs and Cnovs
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                # m.eval()
                for p in m.parameters():
                    p.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        # x = (x - Variable(self.mean)) / Variable(self.std)

        x = self.conv1(x)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 64
        r3 = self.res3(r2)  # 1/8, 128
        r4 = self.res4(r3)  # 1/8, 1024
        r5 = self.res5(r4)
        out = self.pred(r5)
        return out


class VMNetwork(nn.Module):
    def __init__(self, k=20):
        super(VMNetwork, self).__init__()
        self.Encoder = Encoder()
        self.MotionEncoder = MotionEncoderPool()
        self.ResidualEncoder = ResidualEncoder()
        self.WarperConv = WarperConv()
        self.AttentionEncoder = AttentionEncoder()
        self.AttentionOperate = AttentionOperate()
        self.key = k
        self.down = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        # init the conv
        n = self.Conv.kernel_size[0] * self.Conv.kernel_size[1] * self.Conv.out_channels
        self.Conv.weight.data.normal_(0, math.sqrt(2. / n))

        # freeze BNs and Cnovs
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                # m.eval()
                for p in m.parameters():
                    p.requires_grad = False
            if isinstance(m, nn.Conv2d):
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, ref_f, ref_m, key_f, tag_f, motion, residual, motion2, gb):
        # ref and tag and key encoder
        ref_feat = self.Encoder(ref_f)  # [n, 512, h, w]
        tag_feat = self.Encoder(tag_f)  # [n, 512, h, w]
        key_feat = self.Encoder(key_f)  # [n, 512, h, w]
        # motion feature
        m_r3 = self.MotionEncoder(motion)
        # attention map, map is for network and mask is to calculate the loss
        attention_map, attention_mask = self.AttentionEncoder(motion2, gb)
        # residual feature
        res_r3 = self.ResidualEncoder(residual)
        # warp feature
        new_tag = torch.cat([self.WarperConv(key_feat, m_r3), res_r3], dim=1)
        new_tag = self.relu(self.bn(new_tag))
        new_tag = self.Conv(new_tag)
        # attnetion operate
        new_tag = self.AttentionOperate(new_tag, attention_map)
        # process mf and mb
        ref_m = self.down(self.down(self.down(ref_m))) # 1/8
        mf = ref_feat * ref_m # ref_m is [n, 1, h, w]
        mb = ref_feat * (1 - ref_m)
        # claculate the similarity of key image
        key_out = CosineSimilarityCalculate2(key_feat, mf, mb)
        key_out = key_out * 10
        # claculate the similarity of tag image
        tag_out = CosineSimilarityCalculate2(tag_feat, mf, mb)
        tag_out = tag_out * 10
        # claculate the similarity of new tag image
        new_out = CosineSimilarityCalculate2(new_tag, mf, mb)
        new_out = new_out * 10
        return key_out, tag_out, new_out, tag_feat, new_tag, attention_mask

