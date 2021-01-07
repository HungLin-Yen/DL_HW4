# DL_HW4
homework4 for selected topic for visual reconition with deep learning 2020

reference: https://github.com/twtygqyy/pytorch-vdsr

# Guild
Trained using train.h5

Run main_vdsr.py for training


Put test LR_image under 'testing_lr_images/'

Run test.py for testing

# Model
Use the VDSR model implement by https://github.com/twtygqyy/pytorch-vdsr

Trun input image to YCbCr then only process the Y part

Use resize() in PIL.Image for resize input image when testing

# Superparaneter
Initial LR = 0.1,
Step = 10,
gamma = 0.1,
momentum=0.9,
weight decay = 0.0001

Clipping gradients at the threshold 0.4

