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


class TempDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.dir_AB = self.dir_AB + '_temp'
        self.temp_path = self.dir_AB[:-9]+'template/template.jpg'
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        self.size = len(self.AB_paths)
        #self.labels={}
        assert(opt.resize_or_crop == 'resize_and_crop')
        #print(self.dir_dot)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        temp_path = self.temp_path  

        filename = AB_path.split('/')[-1]
        label = [int(filename[0:4])]
        label = torch.LongTensor(label)

        A = Image.open(AB_path).convert('RGB')
        C = Image.open(temp_path)
        w, h = A.size
       
        ori = transforms.ToTensor()(A)
        C = transforms.ToTensor()(C)

        C = C[0, ...]
        C = C.view(1,self.opt.loadSizeY,self.opt.loadSize*10)

        tmp_count=10
        for i in range(0,tmp_count):
            cmbed = torch.cat((ori,C[:,0:h,(i*w):((i+1)*w)]),0)
            if i==0:
                A = cmbed
            else:
                A = torch.cat((A,cmbed),1)

        #print(A.shape,C.shape)
        #A = torch.cat((A,C.view(1,self.opt.loadSize,self.opt.loadSizeY)),0) 
        #B = torch.cat((A,C),0)
        ori = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(ori)
        A = transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))(A)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        return {'A': A, 'B': ori, 'label' : label,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'TempDataset'
