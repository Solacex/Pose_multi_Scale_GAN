import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
from random import randint
from torch.autograd import Variable


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.dir_AB = self.dir_AB + '_14'
        self.dir_dot= self.dir_AB + '_dot'
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        self.dot_paths = sorted(make_dataset(self.dir_dot))
        self.size = len(self.AB_paths)
        #self.labels={}
        assert(opt.resize_or_crop == 'resize_and_crop')
        #print(self.dir_dot)

    def __getitem__(self, index):
        index = randint(0,self.size - 1)

        AB_path = self.AB_paths[index]
        
        dot_path = self.dot_paths[index]
        #print(AB_path,dot_path)
        filename = AB_path.split('/')[-1]
        label = [int(filename[0:4])]
        #print(label)

        label = torch.LongTensor(label)
        #print(label.shape)
        rdm_num=randint(0, 9)
        if rdm_num>4:
            flip=True
        else:
            flip=False
        AB = Image.open(AB_path).convert('RGB')
        C = Image.open(dot_path)
        w, h = AB.size
        w2 = int(w / 2)

        if flip is False:
            A = AB.crop((0, 0, w2, h))#.resize((self.opt.loadSize, self.opt.loadSizeY), Image.BICUBIC)
            B = AB.crop((w2, 0, w, h))#.resize((self.opt.loadSize, self.opt.loadSizeY), Image.BICUBIC)
            C = C.crop((0, 0, w2, h))#.resize((self.opt.loadSize, self.opt.loadSizeY), Image.BICUBIC)
        else:
            B = AB.crop((0, 0, w2, h))#.resize((self.opt.loadSize, self.opt.loadSizeY), Image.BICUBIC)
            A = AB.crop((w2, 0, w, h))#.resize((self.opt.loadSize, self.opt.loadSizeY), Image.BICUBIC)
            C = C.crop((w2, 0, w, h))#.resize((self.opt.loadSize, self.opt.loadSizeY), Image.BICUBIC)
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        C = transforms.ToTensor()(C)
        #A = np.concatenate([A,C],axis=2)
        C = C[0, ...]
        C = C.view(1,self.opt.loadSizeY,self.opt.loadSize)
        #print(A.shape,C.shape)
        #A = torch.cat((A,C.view(1,self.opt.loadSize,self.opt.loadSizeY)),0)

        B = torch.cat((B,C),0)

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5,0.5), (0.5, 0.5, 0.5,0.5))(B)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        
        #print(A.shape)
        return {'A': A, 'B': B, 'label' : label,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
    def get_id(img_path):
        labels = []
        for path, v in img_path:
            filename = path.split('/')[-1]
            label = filename[0:4]
            if label[0:2]=='-1':
                labels.append(-1)
            else:
                labels.append(int(label))
        return labels


