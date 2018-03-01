# coding:utf-8
from pathlib import Path
from skimage import io
from skimage.transform import resize
from Image import NEAREST
import torch
from torchvision import transforms
from torch.utils import data
import numpy as np

class NucleiDetector(data.Dataset):
    def __init__(self, config, phase='train', imgs_trans=None, maks_trans=None):
        '''
        data preprocess
        root contiants all data include train imgs and test imgs
        in __init__() just get each file's path
        '''
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        # transform
        self.imgs_trans = imgs_trans
        self.masks_trans= masks_trans
        # new size
        self.resize_width = config.resize_width
        self.resize_height = config.resize_height
        root = config.root
        if self.phase != 'test':  
            imgs = []
            masks = []       
            # type: generator 
            paths = Path(root+'stage1_train/').glob('*')
            # val/train = 3/10
            # get each img's path and each img's masks folder path
            for p in paths:
                imgs.append(p / 'images' / (p.name+'.png'))
                mask.append(p / 'masks')
            if self.phase == 'train':
                self.imgs = imgs
                self.masks = masks
                self.imgs_num = len(self.imgs)
            # val phase
            else:
                self.imgs = imgs[int(0.7*len(imgs)):]
                self.masks = masks[int(0.7*len(masks)):]
                self.imgs_num = len(self.imgs)
        # test phase
        else:
            # change to list from gen
            self.imgs = list(Path(root+'stage1_test/').glob('*/images/*.png'))
            self.imgs_num = len(imgs)

    def __getitem__(self, index):
        '''
        return this image(3, 128, 128) and its mask(128, 128) and its size
        compose each mask to one
        resize img and mask to self.resize_width * self.resize_height
        '''
        # when using skiamge.io read image size is (w, h, c)
        # img's type: ndarray (width, height, channel)
        img = io.imread(self.imgs[index])
        # just leave the 0 1 2 channel 
        if img.shape[2] > 3:
            assert(img[:, :, 3]!=255).sum()==0
        img = img[:, :, :3]
        before_size = (img.shape[0], img.shape[1])
        if self.imgs_trans is None:
            self.imgs_trans = transforms.Compose([
                transforms.ToPILImage(),
                # Resize image
                transforms.Resize((self.resize_width, self.resize_height), interpolation=NEAREST),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        if self.masks_trans is None:
            self.masks_trans = transforms.Compose([
                transforms.ToPILImage(),
                # Resize image
                transforms.Resize((self.resize_width, self.resize_height), interpolation=NEAREST),
                transforms.ToTensor()
            ])
        # test phase has no mask
        if phase == 'test':
            # img name size
            return self.imgs_trans(img), self.imgs[index].stem, before_size
        # compose mask and trans
        # copy from Tutorial on DSB2018
        else:
            # get this img's mask path object
            # list of all mask file
            mask_files = list(self.masks[index].iterdir())
            masks = np.zeros(len(mask_files), before_size[1], before_size[2])
            for ii, mask in enumerate(mask_files):
                mask = io.imread(mask)
                # just verify mask contain 0 or 255
                # assert (mask[(mask!=0)]==255).all()
                masks[ii] = mask
            
            #tmp_mask = mask.sum(0)
            for ii, mask in enumerate(masks):
                masks[ii] = mask / 255 * (ii+1)
            # mask ndarray
            # ndarray.sum(0) sum the first axis
            # that is this axis disappear other dimension's size doesn't change
            mask = masks.sum(0)
            return self.imgs_trans(img), self.masks_trans(mask)

    def __len__(self):
        return self.imgs_num
