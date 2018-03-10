# coding:utf-8
import warnings

class DefaultConfig(object):
    # data path
    root = '/home/zhengdesheng/DataScienceBowl18/data/data/'
    # save
    model_save_path = '/home/zhengdesheng/DataScienceBowl18/project/model/checkpoints/'
    loss_save_path = '/home/zhengdesheng/DataScienceBowl18/project/utils/saved_loss.csv'
    lr_save_path = '/home/zhengdesheng/DataScienceBowl18/project/utils/saved_lr.csv'
    test_save_path = '/home/zhengdesheng/DataScienceBowl18/data/data/stage1_sample_submission.csv'
    ckpt_path = '/home/zhengdesheng/DataScienceBowl18/project/model/checkpoints/UNet_0310_14:46.ckpt'

    # data prepprocess
    resize = True
    resize_width = 128
    resize_height = 128
    filp = False
    swap = False
    rotate = False
    
    # load data 
    batch_size = 1 # when testing once only load one image for rlencdoing
    num_workers = 8 # how many subprocesses
    pin_memory = True # if True loader will copy tensors in CUDA pinned memory before return them
    
    # train
    use_gpu = True
    lr = 1e-3
    lr_decay = 0.6
    weight_decay = 1e-4
    epoch = 1
    
    # get rle
    threshold = 0.5
    
    
    # if epoch%save_freq == 0: save
    model_save_freq = 5
    

opt = DefaultConfig()
