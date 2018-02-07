# coding:utf-8

class Loss(object):
    def __init__(self, name, loss_save_path):
        super(soft_dice_loss, self).__init__()
        assert isinstance(name, str)
        self.name = name
        self.path = loss_save_path
        self.num = 0
        
    def __call__(self, inputs, targets):
        if name == 'soft_dice_loss':
            l = self.soft_dice_loss(inputs, targets)
        if num%100 == 0:
            self.save_loss(self.num, l, self.path)
        
    def soft_dice_loss(self, inputs, targets):
        # batch size
        num = targets.size(0)
        m1 = inputs.view(num, -1)
        m2 = inputs.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
        score = 1 - score.sum()/num
        return score
        
    def save_loss(self, loss, path):
        # save to csv
        # time,loss_type,loss
        t = time.strftime('%m%d_%H:%M')
        content = [t, self.name, loss]
        with open(path, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(content)
