import numpy as np
import torch
from torch.utils import data
import pickle
import math
import os
from torchvision import transforms as T


def transform():
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    trans = T.Compose([T.ToTensor(), normalize])
    # ToTensor ndarray(hwc) to (chw)
    return trans

class cifar10(data.Dataset):
    def __init__(self, root, transforms=transform(), train=True, test=False):
        self.root = root
        self.transform = transforms
        self.train = train
        self.test = test
        if self.test:
            self.train = False

    def __getitem__(self, item):
        x = math.floor(item / 10000) + 1
        y = item % 10000
        if not self.train and not self.test:
            x = 5
            y = 5000+item

        imgpath = os.path.join(self.root, "data_batch_"+str(x))
        with open(imgpath, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            d_decode = {}
            for k,v in dict.items():
                d_decode[k.decode('utf8')] = v
            dict = d_decode
            data = dict['data'][y]  # 3*32*32==3072
            data = np.reshape(data,(3,32,32))
            data = data.transpose(1,2,0)
            data = self.transform(data)
            label = dict['labels'][y]
            # label = torch.from_numpy(label)

            return data, label


    def __len__(self):
        if self.train:
            return 45000    # train
        elif self.test:
            return 10000    # test
        else:
            return 5000    # valid
