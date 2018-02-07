# coding:utf-8
import torch as t
import time

class BasicModule(t.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path):
        '''
        save and load state dict
        '''
        self.load_state_dict(t.load(path)['state_dict'])

    def save(self, save_dir):
        assert save_dir is not None
        prefix = save_dir + self.model_name 
        name = time.strftime(prefix + '_%m%d_%H:%M.ckpt')
        t.save({'state_dict':self.state_dict()}, name)
        return name + 'saved'
