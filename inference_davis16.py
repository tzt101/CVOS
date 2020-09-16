"""
Copyright (C) University of Science and Technology of China.
Licensed under the MIT License.
"""

import torch
import os
from network.warpCatAtt2_test import VMNetwork
import cv2
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from coviar import load
from coviar import get_num_frames
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.misc import imresize
import time
import torch.nn as nn
import random
from skimage import transform

def l2_norm(x):
    norm = torch.norm(x, 2, 1)
    norm = torch.unsqueeze(norm, dim=1).float()
    #norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()
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

def get_iou(mask, label):
    """
    :param mask: predicted mask with 0 for background and 1 for object
    :param label: label
    :return: iou
    """
    #mask = mask.numpy()
    #label = labels.numpy()
    size = mask.shape
    mask = mask.flatten()
    label = label.flatten()
    m = mask + label
    i = len(np.argwhere(m==2))
    u = len(np.argwhere(m!=0))
    if u == 0:
        u = size[0]*size[1]
    iou = float(i)/u
    if i == 0 and u == 0:
        iou = 1
    return iou

def offsets_to_coordinates(offsets):
    """
    This function change motion vector or optical flow to corresponding coordinates in the image
    :param offsets: tensor with size [h, w, 2], 2 channels denote x (0) and y (1), respectively.
    :return: a tensor with size [h, w, 2], coordinates start from 0
    """
    offsets = torch.Tensor(offsets)

    size = offsets.size()
    h, w, c = size[0], size[1], size[2]

    x_coordinates = torch.Tensor(range(0, w))
    x_coordinates = x_coordinates.unsqueeze(dim=0)
    x_coordinates = x_coordinates.repeat(h, 1)

    y_coordinates = torch.Tensor(range(0, h))
    y_coordinates = y_coordinates.unsqueeze(dim=1)
    y_coordinates = y_coordinates.repeat(1, w)

    offsets[:, :, 0] = -offsets[:, :, 0] + x_coordinates
    offsets[:, :, 1] = -offsets[:, :, 1] + y_coordinates

    return offsets

def coordinates_to_flow_field(coordinates):
    """
    This function shift the coordinates to a flow field
    :param coordinates: tensor with size [h, w, 2], coordinates starts from 0
    :return: a tensor that has the same size with coordinates, contain the flow
            filed. Values: x: -1, y: -1 is the left-top pixel of the input,
            and values: x: 1, y: 1 is the right-bottom pixel of the input.
    """
    coordinates = torch.Tensor(coordinates)
    h, w = coordinates.size()[0], coordinates.size()[1]
    half_h, half_w = h / 2, w / 2

    coordinates[:, :, 0] = (coordinates[:, :, 0]) / half_w # x
    coordinates[:, :, 1] = (coordinates[:, :, 1]) / half_h # y

    return coordinates

def compute_robust_moments(binary_image, isotropic=False):
  index = np.nonzero(binary_image)
  points = np.asarray(index).astype(np.float32)
  if points.shape[1] == 0:
    return np.array([-1.0,-1.0],dtype=np.float32), \
        np.array([-1.0,-1.0],dtype=np.float32)
  points = np.transpose(points)
  points[:,[0,1]] = points[:,[1,0]]
  center = np.median(points, axis=0)
  if isotropic:
    diff = np.linalg.norm(points - center, axis=1)
    mad = np.median(diff)
    mad = np.array([mad,mad])
  else:
    diff = np.absolute(points - center)
    mad = np.median(diff, axis=0)
  std_dev = 1.4826*mad
  std_dev = np.maximum(std_dev, [5.0, 5.0])
  return center, std_dev

def get_gb_image(label, center_perturb = 0.2, std_perturb=0.4, blank_prob=0):
    if not np.any(label) or random.random() < blank_prob:
        #return a blank gb image
        return np.zeros((label.shape))
    center, std = compute_robust_moments(label)
    center_p_ratio = np.random.uniform(-center_perturb, center_perturb, 2)
    center_p = center_p_ratio * std + center
    std_p_ratio = np.random.uniform(1.0 / (1 + std_perturb), 1.0 + std_perturb, 2)
    std_p = std_p_ratio * std
    h,w = label.shape
    x = np.arange(0, w)
    y = np.arange(0, h)
    nx, ny = np.meshgrid(x,y)
    coords = np.concatenate((nx[...,np.newaxis], ny[...,np.newaxis]), axis = 2)
    normalizer = 0.5 /(std_p * std_p)
    D = np.sum((coords - center_p) ** 2 * normalizer, axis=2)
    D = np.exp(-D)
    D = np.clip(D, 0, 1)
    return D


def save_and_calculate_iou(o, save_path, name, label, out_size):
    '''
    :param o: the output of network
    :param save_path: the path to save segmentation results
    :param name: the name of the results
    :param label: the ground truth of this segmentation results
    :param out_size: the output size of the results
    :return: the IoU of the result and the dliated result as the location of next frame
    '''
    output = F.softmax(o, dim=1)
    # output = o
    output = torch.squeeze(output) # [2, new_h, new_w]
    if use_cuda:
        output = output.data.cpu().numpy()
    else:
        output = output.data.numpy()
    output = np.argmax(output, axis=0) # [new_h, new_w]
    gb = output.copy()
    output = np.array(output, dtype=np.uint8)
    di_k = 10
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (di_k, di_k))
    prev_m = cv2.dilate(output, dilate_kernel)
    prev_m = prev_m[np.newaxis, np.newaxis, ...]
    prev_m = np.array(prev_m, dtype=np.float32)
    if use_cuda:
        prev_m = Variable(torch.from_numpy(prev_m).cuda())
    else:
        prev_m = Variable(torch.from_numpy(prev_m))
    # resize
    output = imresize(output, out_size, interp='nearest')
    output[output == 1] = 255
    cv2.imwrite(os.path.join(save_path, name), output)
    output[output == 255] = 1
    iou = get_iou(output, label)

    gb = np.array(gb, dtype=np.float32)
    gb = get_gb_image(gb, center_perturb=0, std_perturb=0)
    gb = gb[np.newaxis, np.newaxis, ...]
    if use_cuda:
        gb = Variable(torch.from_numpy(gb).cuda())
    else:
        gb = Variable(torch.from_numpy(gb))

    return iou, prev_m, gb


# set the params
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
train_file_dir = 'dataset/ImageSets/2016/val.txt'
image_dir = 'dataset/JPEGImages/480p'
label_dir = 'dataset/Annotations/480p'
video_dir = 'dataset/JPEGImages/video/training'
model_dir_cat = 'models/train_0.79_update.pth'

# read the training sequence
seq = []
with open(train_file_dir) as f:
    seq_name = f.readline()
    while seq_name:
        seq.append(seq_name[:-1])
        seq_name = f.readline()

result_save_dirs = 'results/'
im_size = (256, 512)
out_size = (480, 854)
im_mean = (104, 116.67, 122.68)
is_loacte_tag_frame = False
k = 20
use_cuda = True
only_motion = True

# define network
result_save_dir = result_save_dirs
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)
model = VMNetwork()
load_weights(model, torch.load(model_dir_cat))
# torch.cuda.set_device(device=gpu_id)
if use_cuda:
    model.cuda()
model.eval()

# start testing
print('start testing ... ')
IoU = np.zeros((len(seq), 1))
# calculate the forward of different part
for i in range(len(seq)):
    seq_name = seq[i]
    print('start test squence: ', seq_name)
    image_names = os.listdir(os.path.join(image_dir, seq_name))
    image_names = sorted(image_names)
    label_names = os.listdir(os.path.join(label_dir, seq_name))
    label_names = sorted(label_names)

    save_path = os.path.join(result_save_dir, seq_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    seq_iou = 0.0
    for j in range(len(image_names)):
        if j == 0:
            # reference mask
            ref_mask = cv2.imread(os.path.join(label_dir, seq_name, label_names[0]), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(save_path, label_names[0]), ref_mask)
            ref_mask = np.array(imresize(ref_mask, im_size,  interp='nearest'), dtype=np.float32)
            ref_mask[ref_mask == 255] = 1
            gb_mask = get_gb_image(ref_mask.copy(), center_perturb=0, std_perturb=0)
            dliate_prev_mask = np.array(ref_mask.copy(), dtype=np.uint8)
            ref_mask = ref_mask[np.newaxis, np.newaxis, ...] # [n, 1, h, w]
            if use_cuda:
                ref_mask = Variable(torch.from_numpy(ref_mask).cuda())
            else:
                ref_mask = Variable(torch.from_numpy(ref_mask))
            # dilate the mask as the prev mask
            di_k = 10
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (di_k, di_k))
            dliate_prev_mask = cv2.dilate(dliate_prev_mask, dilate_kernel)
            dliate_prev_mask = dliate_prev_mask[np.newaxis, np.newaxis, ...]
            dliate_prev_mask = np.array(dliate_prev_mask, dtype=np.float32)
            if use_cuda:
                prev_mask = Variable(torch.from_numpy(dliate_prev_mask).cuda())
            else:
                prev_mask = Variable(torch.from_numpy(dliate_prev_mask))
            gb_mask = gb_mask[np.newaxis, np.newaxis, ...]
            if use_cuda:
                gb_mask = Variable(torch.from_numpy(gb_mask).cuda())
            else:
                gb_mask = Variable(torch.from_numpy(gb_mask))
            # reference
            ref_image = np.array(imresize(cv2.imread(os.path.join(image_dir, seq_name, image_names[0]), cv2.IMREAD_COLOR), im_size), dtype=np.float32)
            ref_image -= im_mean
            ref_image = ref_image.transpose((2,0,1))
            ref_image = ref_image[np.newaxis, ...]
            if use_cuda:
                ref_image = Variable(torch.from_numpy(ref_image).cuda())
            else:
                ref_image = Variable(torch.from_numpy(ref_image))

            ref_feat = model.Encoder(ref_image)

            # process the mb and mf
            downlabel = model.down(model.down(model.down(ref_mask))) # 1/8
            # set k
            max_k = int(torch.sum(downlabel[0, 0, :]))

            k = min(20, max_k)
            mf = ref_feat * downlabel
            mb = ref_feat * (1 - downlabel)
            # calculate the I frame feature
            key_feat = ref_feat

        else:
            group_i = j // 12
            res_i = j % 12
            label = np.array(cv2.imread(os.path.join(label_dir, seq_name, label_names[j]), cv2.IMREAD_GRAYSCALE), dtype=np.float32)
            label[label == 255] = 1

            if res_i == 0: # I frame
                # key image
                key_image = np.array(imresize(cv2.imread(os.path.join(image_dir, seq_name, image_names[j]), cv2.IMREAD_COLOR), im_size), dtype=np.float32)
                key_image -= im_mean
                key_image = key_image.transpose((2, 0, 1))
                key_image = key_image[np.newaxis, ...]
                if use_cuda:
                    key_image = Variable(torch.from_numpy(key_image).cuda())
                else:
                    key_image = Variable(torch.from_numpy(key_image))
                # forward

                key_feat = model.Encoder(key_image)
                key_out = CosineSimilarityCalculate2(key_feat, mf, mb, k)
                key_out = key_out * 10 * prev_mask
                tmp_iou, prev_mask, gb_mask = save_and_calculate_iou(key_out, save_path, label_names[j], label, out_size)
                seq_iou += tmp_iou

            else: # other frames

                motion_map = np.array(load(os.path.join(video_dir, seq_name, seq_name + '.mp4'), group_i, res_i, 1, True), dtype=np.float32)
                motion_map2 = np.array(load(os.path.join(video_dir, seq_name, seq_name + '.mp4'), group_i, res_i, 1, False), dtype=np.float32)
                resudial_map = np.array(load(os.path.join(video_dir, seq_name, seq_name + '.mp4'), group_i, res_i, 2, True), dtype=np.float32)
                resudial_map -= im_mean
                # process the motion vector
                motion_vector2 = np.array(motion_map2, dtype=np.float32)
                motion_vector2 = motion_vector2.transpose((2, 0, 1))
                motion_vector2 = motion_vector2[np.newaxis, ...]
                if use_cuda:
                    motion_vector2 = Variable(torch.from_numpy(motion_vector2.copy()).cuda())
                else:
                    motion_vector2 = Variable(torch.from_numpy(motion_vector2.copy()))

                motion_map = offsets_to_coordinates(motion_map)
                motion_map = coordinates_to_flow_field(motion_map).numpy()

                motion_map = np.array(motion_map, dtype=np.float32)
                motion_map = motion_map.transpose((2,0,1))
                motion_map = motion_map[np.newaxis, ...]
                if use_cuda:
                    motion_map = Variable(torch.from_numpy(motion_map.copy()).cuda())
                else:
                    motion_map = Variable(torch.from_numpy(motion_map.copy()))
                # process the residual map
                resudial_map = resudial_map.transpose((2,0,1))
                resudial_map = resudial_map[np.newaxis, ...]
                if use_cuda:
                    resudial_map = Variable(torch.from_numpy(resudial_map.copy()).cuda())
                else:
                    resudial_map = Variable(torch.from_numpy(resudial_map.copy()))

                # forward

                # motion feature
                m_r3 = model.MotionEncoder(motion_map)
                # attention map, map is for network and mask is to calculate the loss
                attention_map, attention_mask = model.AttentionEncoder(motion_vector2, gb_mask)
                # resudial feature
                res_r3 = model.ResidualEncoder(resudial_map)
                # warp feature
                tag_feat = model.Conv(model.relu(model.bn(torch.cat([model.WarperConv(key_feat, m_r3), res_r3], dim=1))))
                tag_feat = model.AttentionOperate(tag_feat, attention_map)

                tag_out = CosineSimilarityCalculate2(tag_feat, mf, mb, k)

                tag_out = tag_out * 10
                if is_loacte_tag_frame:
                    tag_out = tag_out * prev_mask
                # save and calculat the iou
                tmp_iou, prev_mask, gb_mask = save_and_calculate_iou(tag_out, save_path, label_names[j], label, out_size)
                seq_iou += tmp_iou
    IoU[i] = seq_iou / (len(image_names) - 1)
    print('the mIoU of ', seq_name, ': ', IoU[i])
print('the average mIoU is ', sum(IoU)/len(seq))






