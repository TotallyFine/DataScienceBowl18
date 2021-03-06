# coding:utf-8

import argparse

from model import UNet
from utils import Loss, train, val, test, save_lr
from config import opt
from data import NucleiDetector

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='train', type=str)
args = parser.parse_args()

def main():
    global args
    net = UNet(3, 1)
    net.load(opt.ckpt_path)
    loss = Loss('soft_dice_loss')
    torch.cuda.set_device(0)
    net = net.cuda()
    loss = loss.cuda()
    
    if args.phase == 'train':
        # train
        dataset = NucleiDetector(opt, phase=args.phase)
        train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=opt.pin_memory)
        lr = opt.lr
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=opt.weight_decay)
        previous_loss = None # haven't run 
        for epoch in range(opt.epoch+1):
            now_loss = train(train_loader, net, loss, epoch, optimizer, opt.model_save_freq, opt.model_save_path)
            if previous_loss is not None and now_loss > previous_loss:
                lr *= opt.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                save_lr(net.model_name, opt.lr_save_path, lr)
            previous_loss = now_loss
    elif args.phase == 'val':
        # val phase
        dataset = NucleiDetector(opt, phase='val')
        val_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=opt.pin_memory)
        val(val_loader, net, loss)   
    else:
        # test phase
        dataset = NucleiDetector(opt, phase='test')
        test_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.num_workers, pin_memory=opt.pin_memory)
        test(test_loader, net, opt)
        
            
if __name__ == '__main__':
    main()
        
        
