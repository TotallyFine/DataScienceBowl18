# coding:utf-8

from skimage.io import concatenate_image
from skimage.transform import resize
from skimage.morphology import label

# resize img and change to one channel

class Resize(object):
    def __init__(self, config):
        self.resize_width = config.resize_width
        self.resize_height = config.resize_height
        
    # 一个问题：使用什么形式的loss 决定了如何处理数据，所以在写代码的时候要先决定输入输出
    def __call__(self, data, masks):
        '''
        data is imgs's list
        masks is masks's list
        '''
        sized_data = []
        for i in range(len(data)):
            # resize 进行插值，param1 是ndarray param2 out_shape
            # preverse_range 是否保留原来的值的类型，否则被转换为float
            img = resize(data[i], (self.resize_height, self.resize_width), mode='contant', preserve_range=True)
        
