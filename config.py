# coding:utf-8
import warnings

class DefaultConfig(object):
    # data path
    root = '/home/jipuzhao/user/Kaggle18/data/'
    # save
    model_save_path = '/home/jipuzhao/user/Kaggle18/project/model/checkpoints/'
    loss_save_path = '/home/jipuzhao/user/Kaggle18/project/utils/saved_loss.csv'
    lr_save_path = '/home/jipuzhao/user/Kaggle18/project/utils/saved_lr.csv'
    test_save_path = '/home/jipuzhao/user/Kaggle18/data/stage1_sample_submission.csv'
    
    # data prepprocess
    resize = True
    resize_width = 128
    resize_height = 128
    filp = False
    swap = False
    rotate = False
    
    # load data
    batch_size = 2 # when testing once only load one image for rlencdoing
    num_workers = 2 # how many subprocesses
    pin_memory = False # if True loader will copy tensors in CUDA pinned memory before return them
    
    # train
    use_gpu = False
    lr = 1e-3
    lr_decay = 0.6
    weight_decay = 1e-4
    epoch = 1
    
    # get rle
    threshold = 0.5
    
    
    # if epoch%save_freq == 0: save
    model_save_freq = 1
    

opt = DefaultConfig()
