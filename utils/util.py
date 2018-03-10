# coding:utf-8
import time 
import csv
from torch.autograd import Variable
from torch import squeeze, cat
import numpy as np
from skimage.transform import resize #skimage's resize only transfrom int type 
from skimage import io

def train(data_loader, net, loss, epoch, optimizer, save_freq, save_path):
    # set net to train model, Dropout and BN will work
    net.train()
    
    
    # data (3, 128, 128) target (1, 128, 128) size(channel, width, height)
    for data, target in data_loader:
        data = Variable(data.cuda(async=True))
        target = Variable(target.cuda(async=True))
        # output is (1, 128, 128)
        output = net(data)
        # now output is (128, 128)
        output = squeeze(output, dim=0)
        loss_output = loss(output, target)
        if epoch % 5 == 0:
            print(loss_output)
            #save_loss(loss_output, loss_path)
        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step()
    
    if epoch % save_freq == 0:
        net.save(save_path)
        
def val(data_loader, net, loss):
    # set eval dropout batchNorm will not work
    net.eval()
    for data, target in data_loader:
        data = Variable(data.cuda(async=True))
        target = Variable(target.cuda(async=True))
        output = net(data)
        output = squeeze(output, dim=0)
        loss_output = loss(output, target)
        
        print(loss_output)
        
def test(data_loader, net, opt):
    # when test once load one img
    net.eval()
    # size is (width, height) 
    for data, name, w, h in data_loader:
        #print(w) pytorch change number to tensor,but not change list to tensor only change number in list to tensor
        size = cat((w, h)).numpy().astype(np.int32)
        data = Variable(data.cuda(async=True))
        # output is (1, 128, 128)
        output = net(data)
        output = squeeze(output, dim=0)
        if opt.use_gpu:
            output = output.data.cpu().numpy()
        else:
            output = output.data.numpy()
        binary = output > opt.threshold # 0.5
        binary = binary.astype(np.int32)
        #assert isinstance(size[0], int)
        binary = binary.reshape((128, 128))
        print(binary.shape, binary.dtype)
        io.imsave('/home/zhengdesheng/DataScienceBowl18/data/data/binary.png', binary)
        img = resize(binary, size, mode='constant', preserve_range=True)
        io.imsave('/home/zhengdesheng/DataScienceBowl18/data/data/img.png', img)
        rle = rle_encoding(img)
        # save into csv file
        content = [name, rle]
        with open(opt.test_save_path, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(content)
        print(name, 'done')
        break  
def rle_encoding(x):
    # get one img's run-length encoding   
    # x array of shape (height, width) 1-mask 0-background
    # return run length as list
    # .T set Fortran order down-then-right
    dots = np.where(x.T.flatten()==1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(i) for i in run_lengths])

def save_lr(model, path, lr):
    # save to csv
    # model,time,lr
    t = time.strftime('%m%d_%H:%M')
    content = [model, t, lr]
    with open(path, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(content)

def save_loss(loss, path):
    # save to csv
    # time, loss_name, lossvar
    t = time.strftime('%m%d_%H:%M')
    content = [t, 'soft_dice_loss', loss]
    with open(path, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(content)
