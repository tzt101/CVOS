"""
Copyright (C) University of Science and Technology of China.
Licensed under the MIT License.
"""

import os
import torch
from data import davis16
from network import warpCatAtt2
from torchvision import transforms
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pdb
import torch.nn as nn
import pydensecrf


def load_weights(cnn_model, weights):
    """
    argus:
    :param cnn_model: the cnn networks need to load weights
    :param weights: the pretrained weigths
    :return: no return
    """
    pre_dict = cnn_model.state_dict()
    for key, val in weights.items():
        if key[0:7] == 'module.':  # the pretrained networks was trained on multi-GPU
            key = key[7:]  # remove 'module.' from the key
        if key in pre_dict.keys():
            if isinstance(val, Parameter):
                val = val.data
            pre_dict[key].copy_(val)
    cnn_model.load_state_dict(pre_dict)


def tr_get_miou(outputs, labels):
    batch_size = outputs.size()[0]
    soft = torch.nn.Softmax()
    outputs = soft(outputs)
    outputs = outputs.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    iou = 0
    for i in range(batch_size):
        pred = outputs[i, 1, :, :]
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        gt = labels[i, ...]
        # print(np.unique(pred), np.unique(gt))
        iou += get_iou(pred, gt)
    iou /= batch_size
    return iou


def get_iou(mask, label):
    """
    :param mask: predicted mask with 0 for background and 1 for object
    :param label: label
    :return: iou
    """
    # mask = mask.numpy()
    # label = labels.numpy()
    size = mask.shape
    mask = mask.flatten()
    label = label.flatten()
    m = mask + label
    i = len(np.argwhere(m == 2))
    u = len(np.argwhere(m != 0))
    if u == 0:
        u = size[0] * size[1]
    iou = float(i) / u
    if i == 0 and u == 0:
        iou = 1
    return iou


def wieghted_CrossEntropyLoss(output, label):
    # calculate the weight
    num_labels_pos = torch.sum(label).float()
    num_labels_neg = torch.sum(1.0 - label).float()
    num_total = num_labels_pos + num_labels_neg
    w = torch.ones(2).float()
    w[0] = num_labels_pos / num_total  # the class 0 is the background
    w[1] = num_labels_neg / num_total  # the class 1 is the object
    w = Variable(w.cuda())
    # define the loss
    loss_function = torch.nn.CrossEntropyLoss(weight=w)
    loss = loss_function(output, label)
    return loss


def feat_loss(f_t, f_tag):
    n, c, h, w = f_t.shape
    f_t = f_t.view(n, c, -1)
    f_tag = f_tag.view(n, c, -1)
    loss = torch.norm(f_t - f_tag, p=2, dim=2).mean()
    return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


def get_encoder_decoder_params(model):
    b = []
    b.append(model.module.Encoder)
    b.append(model.module.Decoder)
    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_residual_params(model):
    b = []
    b.append(model.module.ResidualEncoder)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


#  param of training
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
save_model_dir = 'models/'
prev_model_dir = 'models/train_VM.pth'
batch_size = 4
learning_rate = 1e-6
epoch = 60
each_show_iter = 10

# load the model
print('define model...')
model = warpCatAtt2.VMNetwork()
load_weights(model, torch.load(prev_model_dir))
# load_weights(model, torch.load(prev_model_dir2))
model = torch.nn.DataParallel(model)
model.cuda()
print('The number of GPU is', torch.cuda.device_count())

# training optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)
optimizer.zero_grad()

# define loss
loss_function = torch.nn.CrossEntropyLoss()
dice_loss_function = DiceLoss()

# loading the training data
train_data = davis16.Davis16DataSet()

# start training
print('start training ... ')

for epoch_index in range(epoch):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    num_img_tr = len(train_loader)
    # one training epoch
    for iter, sample in enumerate(train_loader):
        ref_img, ref_mask = Variable(sample["ref_frame"].cuda()), Variable(sample["ref_mask"].cuda())
        key_img, key_label = Variable(sample["key_frame"].cuda()), Variable(sample["key_label"].cuda())
        motion, residual, tag_label, tag_img = Variable(sample["motion"].cuda()), Variable(sample["resudial"].cuda()), \
                                               Variable(sample["tag_label"].cuda()), Variable(
            sample["tag_frame"].cuda())
        motion2 = Variable(sample["motion2"].cuda())
        gb = Variable(sample["gb_mask"].cuda())
    
        # forward
        key_out, t_out, tag_out, tag_feat, new_tag, att_out = model.forward(ref_img, ref_mask, key_img, tag_img, motion,
                                                                   residual, motion2, gb)
    
        # loss
        tag_loss = wieghted_CrossEntropyLoss(tag_out, tag_label)
        key_loss = wieghted_CrossEntropyLoss(key_out, key_label)
        att_loss = wieghted_CrossEntropyLoss(att_out, tag_label)
        f_loss = feat_loss(tag_feat, new_tag)
        # print(tag_loss.data.cpu().numpy(), key_loss.data.cpu().numpy())
        loss = (key_loss + tag_loss + 0.5*att_loss) / 2.5
        # mIoU
        tag_iou = tr_get_miou(tag_out.clone(), tag_label.clone())
        key_iou = tr_get_miou(key_out.clone(), key_label.clone())
        t_iou = tr_get_miou(t_out.clone(), tag_label.clone())
    
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
        # print
        if (iter + 1) % each_show_iter == 0:
            print('0epoch = ', epoch_index + 1, 'iter = ', iter + 1, 'loss = %.6f' % loss.data.cpu().numpy(),
                  'tag_mIoU = %.4f' % (tag_iou), 't_mIoU = %.4f' % (t_iou), 'key_mIoU = %.4f' % (key_iou),
                  'key_loss = %.6f' % key_loss.data.cpu().numpy(), 'tag_loss = %.6f' % tag_loss.data.cpu().numpy(),
                  'att_loss = %.6f' % att_loss.data.cpu().numpy())


    # save the model
    if (epoch_index + 1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(save_model_dir, 'train_' + str(epoch_index + 1) + '.pth'))






