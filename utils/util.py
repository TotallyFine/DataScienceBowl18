# coding:utf-8
import time
from torch.autograd import Variable
from torch import squeeze, where
from skimage.transform import resize


def train(data_loader, net, loss, epoch, optimizer, save_freq, save_path):
    # set net to train model, Dropout and BN will work
    net.train()
    
    # data (3, 128, 128) target (128, 128) size(channel, width, height)
    for data, target in data_loader:
        data = Variable(data)
        target = Variable(target)
        # output is (1, 128, 128)
        output = net(data)
        # now output is (128, 128)
        output = squeeze(output, dim=0)
        loss_output = loss(output, target)
        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step()
    
    if epoch % save_freq == 0:
        net.save(save_path)
        
def val(data_loader, net, loss):
    # set eval dropout batchNorm will not work
    net.eval()
    
    for data, target in data_loader:
        data = Variable(data)
        target = Variable(target)
        output = net(data)
        output = squeeze(output, dim=0)
        loss_output = loss(output, target)
        print('val loss', loss_output)
        
def test(data_loader, net, opt):
    # when test once load one img
    net.eval()
    # size is (height, width) 
    for data, name, size in data_loader:
        data = Variable(data)
        # output is (1, 128, 128)
        output = Variable(data)
        output = squeeze(output, dim=0) 
        binary = output> opt.threshold # 0.5
        img = resize(binary, size, mode='constant', preserve_range=True)
        rle = rle_encoding(img)
        # save into csv file
        content = [name, rle]
        with open(opt.test_save_path, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(content)
          
def rle_encoding(x):
    # get one img's run-length encoding   
    # x array of shape (height, width) 1-mask 0-background
    # return run length as list
    # .T set Fortran order down-then-right
    dots = where(x.T.flatten()==1)[0]
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
