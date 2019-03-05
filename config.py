import torch
import os
class DefaultConfig(object):
    insize = 3 * 32 * 32
    outsize = 10
    epoch = 100
    batchsize = 64
    numworker = 4
    lr = 0.001
    weight_decay = 0e-5
    printinter = 20
    root = 'cifar-10-batches-py'
    # load_model_path = os.path.join('snapshot','simNN_0_0305_165425.pth')
    load_model_path = None
    train = True
    use_cuda = torch.cuda.is_available()

    def parse(self,kwargs):
        for k, v in kwargs.item():
            setattr(self, k, v)
