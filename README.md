# CVOS
REAL TIME COMPRESSED VIDEO OBJECT SEGMENTATION (ICME 2019)

## Abstract
Video object segmentation is a challenging task with wide variety of applications. Although recent CNN based methods have achieved great performance, they are far from being applicable for real time applications. In this paper, we propose a propagation based video object segmentation method in compressed domain to accelerate inference speed. We only extract features from I-frames by the traditional deep segmentation network. And the features of P-frames are propagated from I-frames. Apart from feature warping, we propose two
effective modules in the process of feature propagation to ensure the representation ability of propagated features in terms of appearance and location. Residual supplement module is used to supplement appearance information lost in warping, and spatial attention module mines accurate spatial saliency prior to highlight the specified object. Compared with recent state-of-the-art algorithms, the proposed method achieves comparable accuracy while much faster inference speed.

## Installation
Clone this repo.
```bash
git clone https://github.com/tzt101/CVOS.git
cd CVOS/
```

This code is tested on PyTorch 0.4.0. You need to install some necessary dependencies and [coviar](https://github.com/chaoyuaw/pytorch-coviar.git).

## Dataset and Pretrained Model
You can download the prepared dataset and pretrained models from [OneDrive](). Please unzip it in the folder CVOS/. If you want to generate new video dataset, please refer to [coviar](https://github.com/chaoyuaw/pytorch-coviar.git).

## Inference with pretrained model
```bash
python inference.py
```
You can set the inference parameters on inference.py (line 229~251).

## Training New Models
```bash
python train.py
```
You can set the training parameters on train.py (line 149~155) and dataset parameters on data/davis16.py (line 134~140).

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{tan2019real,
  title={Real Time Compressed Video Object Segmentation},
  author={Tan, Zhengtao and Liu, Bin and Li, Weihai and Yu, Nenghai},
  booktitle={2019 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={628--633},
  year={2019},
  organization={IEEE}
}
```
```
@article{tan2020real,
  title={Real Time Video Object Segmentation in Compressed Domain},
  author={Tan, Zhentao and Liu, Bin and Chu, Qi and Zhong, Hangshi and Wu, Yue and Li, Weihai and Yu, Nenghai},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2020},
  publisher={IEEE}
}
```



