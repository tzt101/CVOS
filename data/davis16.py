"""
Copyright (C) University of Science and Technology of China.
Licensed under the MIT License.
"""

import os
from scipy.misc import imresize
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np
import random
import data.custom_transforms as tr
from torchvision import transforms
import torch
from coviar import get_num_frames
from coviar import load
from skimage import transform

def erode_dliate(img):
    if random.random() < 0.5:
        return img
    er_k = int(10*random.random()) + 1
    di_k = int(10*random.random()) + 1
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(er_k, er_k))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(di_k, di_k))
    is_er_first = random.random()
    if is_er_first < 0.5:
        img_f = cv2.erode(img, erode_kernel)
        # if random.random() < 0.5:
        #     img_f = cv2.dilate(img_f, dilate_kernel)
        return img_f
    else:
        img_f = cv2.dilate(img, dilate_kernel)
        # if random.random() < 0.5:
        #     img_f = cv2.erode(img_f, erode_kernel)
        return img_f

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

def mot_resize(mot, s_size=(256,512), b_size=(480,864)):
    # mot
    mot_s = np.array(transform.resize(mot, s_size, mode='constant'), dtype=np.float32)
    mot_s[..., 0] = mot_s[..., 0] / b_size[1] * s_size[1]
    mot_s[..., 1] = mot_s[..., 1] / b_size[0] * s_size[0]
    return mot_s

def res_resize(res, s_size=(256,512), b_size=(480,864)):
    # res
    res_s = np.array(transform.resize(res/255., s_size,mode='constant')*255, dtype=np.float32)
    return res_s

class Davis16DataSet(Dataset):
    def __init__(self):
        self.train_file_dir = 'datset/ImageSets/2016/train.txt'
        self.video_dir = 'dataset/JPEGImages/video/training'
        self.image_dir = 'dataset/JPEGImages/480p'
        self.label_dir = 'dataset/Annotations/480p_all'
        self.b_size = (480, 864)
        self.im_size = (256, 512)
        self.im_mean = (104, 116.67, 122.68)
        self.train_files = []
        self.video_transforms = transforms.Compose([tr.ToTensor()])
        self.key_tag_transforms = transforms.Compose([tr.ScaleNRotate(rots=(-10, 10), scales=(1, 1)), tr.RandomMove(0.01)])
        self.label_transforms = transforms.Compose([tr.ScaleNRotate(rots=(-5, 5),scales=(1, 1)), tr.RandomMove(0.01), tr.ToTensor()])
        self.image_transforms = transforms.Compose([tr.RandomHorizontalFlip(), tr.ScaleNRotate(rots=(-30, 30), scales=(0.75, 1.25)), tr.RandomMove(0.05),tr.ToTensor()])

        # read the training sequence of DAVIS 2016
        train_seq_names = []
        with open(self.train_file_dir) as f:
            seq_name = f.readline()
            while seq_name:
                train_seq_names.append(seq_name[:-1])
                seq_name = f.readline()


        # generate the training files
        for i in range(len(train_seq_names)):
            seq_name = train_seq_names[i]
            video_path = os.path.join(self.video_dir, seq_name, seq_name + '.mp4')
            image_names = sorted(os.listdir(os.path.join(self.image_dir, seq_name)))
            seq_mun = get_num_frames(video_path)
            group_mun = seq_mun // 12
            final_idx = seq_mun % 12
            for g in range(group_mun):
                for idx in range(1, 12):
                    ref_img_name = random.sample(image_names, 1)
                    # if the reference mask contain the object
                    ref_m = np.array(cv2.imread(
                            os.path.join(self.label_dir, seq_name, ref_img_name[0][:-4] + '.png'),
                            cv2.IMREAD_GRAYSCALE), dtype=np.float32)
                    if ref_m.max() > 0:
                        coord = np.where(ref_m == 255)
                        h_min = np.min(coord[0])
                        h_max = np.max(coord[0])
                        w_min = np.min(coord[1])
                        w_max = np.max(coord[1])
                        if (h_max - h_min) * (w_max - w_min) > 40:
                            tag_img_idx = g*12 + idx
                            key_img_idx = g*12
                            self.train_files.append({
                                    "image_dir": self.image_dir,
                                    "label_dir": self.label_dir,
                                    "seq_name": seq_name,
                                    "key_frame": image_names[key_img_idx],
                                    "key_label": image_names[key_img_idx][:-4] + '.png',
                                    "ref_frame": ref_img_name[0],
                                    "ref_mask": ref_img_name[0][:-4] + '.png',
                                    "tag_idx": tag_img_idx,
                                    "tag_frame": image_names[tag_img_idx],
                                    "tag_label": image_names[tag_img_idx][:-4] + '.png',
                                    "prev_label": image_names[tag_img_idx-1][:-4] + '.png',
                                    "video_path": video_path
                            })
            if final_idx >= 2:
                for idx in range(1, final_idx):
                    ref_img_name = random.sample(image_names, 1)
                    # if the reference mask contain the object
                    ref_m = np.array(
                            cv2.imread(os.path.join(self.label_dir, seq_name, ref_img_name[0][:-4] + '.png'),
                                   cv2.IMREAD_GRAYSCALE), dtype=np.float32)
                    if ref_m.max() > 0:
                        coord = np.where(ref_m == 255)
                        h_min = np.min(coord[0])
                        h_max = np.max(coord[0])
                        w_min = np.min(coord[1])
                        w_max = np.max(coord[1])
                        if (h_max - h_min) * (w_max - w_min) > 40:
                            tag_img_idx = group_mun*12 + idx
                            key_img_idx = group_mun*12
                            self.train_files.append({
                                    "image_dir": self.image_dir,
                                    "label_dir": self.label_dir,
                                    "seq_name": seq_name,
                                    "key_frame": image_names[key_img_idx],
                                    "key_label": image_names[key_img_idx][:-4] + '.png',
                                    "ref_frame": ref_img_name[0],
                                    "ref_mask": ref_img_name[0][:-4] + '.png',
                                    "tag_idx": tag_img_idx,
                                    "tag_frame": image_names[tag_img_idx],
                                    "tag_label": image_names[tag_img_idx][:-4] + '.png',
                                    "prev_label": image_names[tag_img_idx - 1][:-4] + '.png',
                                    "video_path": video_path
                            })




    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, item):
        train_files = self.train_files[item]

        # process the reference input
        ref_frame = cv2.imread(os.path.join(train_files["image_dir"], train_files["seq_name"], train_files["ref_frame"]), cv2.IMREAD_COLOR)
        ref_mask = cv2.imread(os.path.join(train_files["label_dir"], train_files["seq_name"], train_files["ref_mask"]), cv2.IMREAD_GRAYSCALE)
        # resize the image and mask
        ref_frame = imresize(ref_frame, self.im_size)
        ref_mask = imresize(ref_mask, self.im_size, interp='nearest')
        # sub mean
        ref_frame = np.array(ref_frame, dtype=np.float32)
        ref_frame -= self.im_mean
        # set the 255 value of mask to 1
        ref_mask = np.array(ref_mask, dtype=np.float32)
        ref_mask = ref_mask / np.max([ref_mask.max(), 1e-8])
        # transformation
        ref_sample = {'image': ref_frame, 'gt': ref_mask}
        ref_sample = self.image_transforms(ref_sample)

        # process the key input
        key_frame = cv2.imread(os.path.join(train_files["image_dir"], train_files["seq_name"], train_files["key_frame"]), cv2.IMREAD_COLOR)
        key_label = cv2.imread(os.path.join(train_files["label_dir"], train_files["seq_name"], train_files["key_label"]), cv2.IMREAD_GRAYSCALE)
        # resize the image and mask
        key_frame = imresize(key_frame, self.im_size)
        key_label = imresize(key_label, self.im_size, interp='nearest')
        # sub mean
        key_frame = np.array(key_frame, dtype=np.float32)
        key_frame -= self.im_mean
        # set the 255 value of mask to 1
        key_label = np.array(key_label, dtype=np.float32)
        key_label = key_label / np.max([key_label.max(), 1e-8])


        # process the tag input: motion vector, resudial map and label and previous mask
        group = train_files["tag_idx"] // 12
        index = train_files["tag_idx"] % 12
        motion_vector = load(train_files["video_path"], group, index, 1, True)
        motion_vector2 = load(train_files["video_path"], group, index, 1, False)
        resudial = np.array(load(train_files["video_path"], group, index, 2, True), dtype=np.float32)
        tag_label = cv2.imread(os.path.join(train_files["label_dir"], train_files["seq_name"], train_files["tag_label"]),cv2.IMREAD_GRAYSCALE)
        prev_mask = cv2.imread(os.path.join(train_files["label_dir"], train_files["seq_name"], train_files["prev_label"]),cv2.IMREAD_GRAYSCALE)
        tag_frame = cv2.imread(os.path.join(train_files["image_dir"], train_files["seq_name"], train_files["tag_frame"]), cv2.IMREAD_COLOR)
        # resize the mask
        tag_frame = imresize(tag_frame, self.im_size)
        tag_label = imresize(tag_label, self.im_size, interp='nearest')
        prev_mask = imresize(prev_mask, self.im_size, interp='nearest')
        # # resize the motion and residual
        # resudial = res_resize(resudial)
        # motion_vector = mot_resize(motion_vector)
        # motion_vector2 = mot_resize(motion_vector2)
        # sub mean
        resudial -= self.im_mean
        tag_frame = np.array(tag_frame, dtype=np.float32)
        tag_frame -= self.im_mean
        # set the 255 value of mask to 1
        tag_label = np.array(tag_label, dtype=np.float32)
        tag_label = tag_label / np.max([tag_label.max(), 1e-8])
        prev_mask = np.array(prev_mask, dtype=np.float32)
        prev_mask = prev_mask / np.max([prev_mask.max(), 1e-8])

        # previous transformation of key and tag
        motion_max = abs(motion_vector).max()
        motion_vector = motion_vector / (motion_max + 1)
        motion_max2 = abs(motion_vector2).max()
        motion_vector2 = motion_vector2 / (motion_max2 + 1)
        prev_sample = {'key_image': key_frame.copy(), 'key_label': key_label.copy(),
                       'tag_motion': motion_vector.copy(),'tag_motion2': motion_vector2.copy(),
                       'tag_res': resudial.copy(), 'tag_label': tag_label.copy(), 'tag_frame': tag_frame.copy(), 'prev_mask': prev_mask.copy()}

        prev_sample = self.key_tag_transforms(prev_sample)

        key_frame = prev_sample['key_image']
        key_label = prev_sample['key_label']
        motion_vector = prev_sample['tag_motion'] * (motion_max + 1)
        motion_vector2 = prev_sample['tag_motion2'] * (motion_max2 + 1)
        resudial = prev_sample['tag_res']
        tag_label = prev_sample['tag_label']
        tag_frame = prev_sample['tag_frame']
        prev_mask = prev_sample['prev_mask']


        # key input transformation
        key_sample = {'image': key_frame, 'gt': key_label}
        key_sample = self.video_transforms(key_sample)
        final_key_label = key_sample['gt'].long()
        final_key_label = torch.squeeze(final_key_label)

        # tag input transformation
        tag_sample = {'image': tag_frame}
        tag_sample = self.video_transforms(tag_sample)
        motion_vector2 = np.array(motion_vector2, dtype=np.float32)
        motion_vector2 = motion_vector2.transpose((2,0,1))
        motion_vector2 = torch.from_numpy(motion_vector2.copy())
        motion_vector = offsets_to_coordinates(motion_vector)
        motion_vector = coordinates_to_flow_field(motion_vector).numpy()
        motion_vector = np.array(motion_vector, dtype=np.float32)
        motion_vector = motion_vector.transpose((2,0,1))
        motion_vector = torch.from_numpy(motion_vector.copy())
        resudial = resudial.transpose((2,0,1))
        resudial = torch.from_numpy((resudial.copy()))
        tag_label = torch.from_numpy(tag_label)
        tag_label = tag_label.long()

        gb_mask = get_gb_image(prev_mask)
        gb_mask = gb_mask[np.newaxis, ...]
        gb_mask = torch.from_numpy(gb_mask)




        data = {
            "ref_frame": ref_sample['image'],
            "ref_mask": ref_sample['gt'],
            "key_frame": key_sample['image'],
            "key_label": final_key_label,
            "motion": motion_vector,
            "resudial": resudial,
            "tag_label": tag_label,
            "tag_frame": tag_sample['image'],
            "motion2": motion_vector2,
            "gb_mask": gb_mask
        }

        return data





