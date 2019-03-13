import torch
from torch.utils.data import DataLoader
import torchvision
import datasetmaker
import model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import DefaultConfig
import time
import os
from torch.nn import init


opt = DefaultConfig()

def save(model,epoch):
    prefix = os.path.join('snapshot','simNN_'+str(epoch)+'_')
    # prefix = 'snapshot'+ os.sep+'simNN_'+str(epoch)+'_'
    name = time.strftime(prefix + '%m%d_%H%M%S.pth')
    torch.save(model.state_dict(),name)


def plotlc(x, y, figname='learning_curve'):
    plt.plot(x, y)
    plt.title('learning curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.show()
    plt.savefig(figname)


def train(model, trainloader):
    if opt.use_cuda:
        model = model.cuda()
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    # optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=opt.weight_decay)
    weight_p = []
    bias_p = []
    for name, para in model.named_parameters():
        if 'bias' in name:
            bias_p += [para]
        else:
            weight_p += [para]
    optimizer = torch.optim.Adam([{'params': weight_p, 'weight_decay': opt.weight_decay},
                                  {'params': bias_p, 'weight_decay': 0}], lr=lr)
    previous_loss = 1e10
    pEpoch = []
    pLoss = []

    for epoch in range(opt.epoch):
        loss_all = 0
        total_accuracy = 0
        for i, (input, target) in enumerate(trainloader):
            if opt.use_cuda:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            pred = torch.max(score, 1)[1]
            accuracy = float((pred == target).sum())
            accuracy = accuracy * 100 / input.size(0)
            # print((pred == target).sum(dim=0,keepdim=False))

            total_accuracy += accuracy
            loss_all += float(loss)

            if i % opt.printinter ==0:
                print("Epoch: ", epoch, "| Iter:", i, "| Loss:", float(loss), "| Accuracy:", accuracy, "%")

        avgloss = loss_all / len(trainloader)
        avgaccuracy = total_accuracy / len(trainloader)
        print("the end of Epoch: ", epoch, "| AVGLoss:", avgloss, "| Accuracy:", avgaccuracy, "%")
        save(model, epoch)

        # plot
        pEpoch.append(epoch)
        pLoss.append(avgloss)
        plotlc(pEpoch, pLoss)

        # val the model
        # validset = datasetmaker.cifar10(opt.root, train=False, test=False)
        # validloader = DataLoader(validset, batch_size=opt.batchsize, shuffle=False, num_workers=opt.numworker)
        # valaccuracy = val(model, validloader)
        # print("validation of Epoch: ", epoch, "| Accuracy:", valaccuracy, "%")

        # update lr
        # if avgloss > previous_loss:
        #     lr = lr * opt.lr_decay
        #     for para in optimizer.param_groups:
        #         para['lr'] = lr
        #     print("learning rate changes to ",lr)
        # previous_loss = avgloss


def val(model, valloader):
    model.eval()
    accuracy = 0
    avgcount = 0
    for i, (input, target) in enumerate(valloader):
        if opt.use_cuda:
            input = input.cuda()
            target = target.cuda()
        score = model(input)
        pred = torch.max(score, 1)[1]
        accuracy += float((pred == target).sum())
        avgcount += input.size(0)
    accuracy = accuracy * 100 / avgcount
    model.train()
    return accuracy


# it's exactly the same as val function
def test(model, testloader):
    model.eval()
    if opt.use_cuda:
        model = model.cuda()
    avgcount = 0
    accuracy = 0
    for i, (input, target) in enumerate(testloader):
        if opt.use_cuda:
            input = input.cuda()
            target = target.cuda()
        score = model(input)
        pred = torch.max(score, 1)[1]
        accuracy += float((pred == target).sum())
        avgcount += input.size(0)
    accuracy = accuracy * 100 / avgcount
    # im not sure whether need model.train() or not
    model.train()
    return accuracy


def main():
    if opt.train:
        # trainset = datasetmaker.cifar10(opt.root)
        trainset = torchvision.datasets.CIFAR10(root='.', train=True, download=False,
                                                transform=datasetmaker.transform())
    else:
        # trainset = datasetmaker.cifar10(opt.root, train=False, test=True)
        trainset = torchvision.datasets.CIFAR10(root='.', train=False, download=False,
                                                transform=datasetmaker.transform())

    # a=trainset[20000][0]
    # print(a.size())
    # print(len(trainset))
    # trainset = torchvision.datasets.CIFAR10(root='.', train=True, download=False, transform=datasetmaker.transform())
    # b=trainset[20000][0]
    # print(b.size())
    # print((a==b).sum())

    trainloader = DataLoader(trainset, batch_size=opt.batchsize, shuffle=True, num_workers=opt.numworker)

    simNN = model.simNN(opt.insize, opt.outsize)
    # initialize
    # for name, params in simNN.named_parameters():
    #     if name.find('linear') != -1:
    #         if name.find('weight') != -1:
    #             init.kaiming_normal(params)  # weight
    #         # init.kaiming_normal(params[0])  # weight
    #         # init.xavier_normal(params[1])  # bias


    if opt.load_model_path:
        simNN.load_state_dict(torch.load(opt.load_model_path))
        print("Load Success!", opt.load_model_path)
    if opt.train:
        train(simNN, trainloader)
    else:
        test(simNN, trainloader)


if __name__ == "__main__":
    main()
