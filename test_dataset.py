import os
import torch
import numpy
from torch.utils.data import Dataset
from imageio import imread
from PIL import Image
from torch.autograd import Variable

class SR_dataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.test_names = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]

    def __len__(self):
        return len(self.test_names)

    def __getitem__(self, idx):
        #load image info
        img_name = self.test_names[idx]

        #open image
        img_path = os.path.join(self.root, img_name)
        image_YCbCr = imread(img_path, pilmode="YCbCr")
        shape = image_YCbCr.shape
        image_YCbCr = numpy.array(Image.fromarray(image_YCbCr).resize(size=(shape[1] * 3, shape[0] * 3)))

        image_Y = image_YCbCr[:,:,0].astype(float)
        image = image_Y/255
        image = Variable(torch.from_numpy(image).float()).view(1, -1, image.shape[0], image.shape[1])

        return img_name, image, image_YCbCr