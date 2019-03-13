import torch
import os
class DefaultConfig(object):
    insize = 3 * 32 * 32
    outsize = 10
    epoch = 600
    batchsize = 256
    numworker = 0
    lr = 0.001
    # weight_decay = 0e-5
    weight_decay = 1e-5
    lr_decay = 0.5
    printinter = 20
    root = 'cifar-10-batches-py'
    # load_model_path = os.path.join('snapshot','simNN_0_0305_165425.pth')

    # is u wanna test, modify the load_model_path and train below
    load_model_path = None
    train = True
    use_cuda = torch.cuda.is_available()

    def parse(self,kwargs):
        for k, v in kwargs.item():
            setattr(self, k, v)
