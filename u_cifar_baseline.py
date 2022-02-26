from __future__ import print_function

#######################################################################################
# Run Command : 
# python u_cifar_baseline.py --epoch 20  
#######################################################################################
import argparse
import os
import random
import shutil
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataloader import ChestXrayDataSet, u_ChestXrayDataSet
from torchvision import datasets, models, transforms

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, compute_AUCs
import matplotlib.pyplot as plt

# ############################### Parameters ###############################
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[5, 10, 15],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' (default: resnet18)')
parser.add_argument('--activation', type=str, default="relu", help='Activation Function to use')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=1, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--save_dir', default='u_results/0/', type=str)
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'


use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


DATA_DIR = "/data/NIH_Xray/images/"

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    os.makedirs(args.save_dir, exist_ok=True)

    # ############################### Dataset ###############################
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    
    CLASS_NAMES = [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia']
    num_classes = len(CLASS_NAMES)
    
    
    # #################### train, dev, test split ####################
    train_dataset = ChestXrayDataSet(data_dir = DATA_DIR, mode = "train", transform = transform_train)
    dev_dataset = ChestXrayDataSet(data_dir = DATA_DIR, mode = "val", transform = transform_train)
    test_dataset = ChestXrayDataSet(data_dir = DATA_DIR, mode = "test", transform = transform_test)
    
    u_train_dataset = u_ChestXrayDataSet(data_dir = DATA_DIR, mode = "train", transform = transform_train)

    
    print('Total image in train, ', len(train_dataset))
    print('Total image in valid, ', len(dev_dataset))
    print('Total image in test, ', len(test_dataset))
    print('Total image in unlabelled set, ', len(u_train_dataset))
    

    trainloader = data.DataLoader(train_dataset, batch_size=args.train_batch,
                                  shuffle=True, num_workers=args.workers)

    devloader = data.DataLoader(dev_dataset, batch_size=args.train_batch,
                                shuffle=False, num_workers=args.workers)

    testloader = data.DataLoader(test_dataset, batch_size=args.test_batch,
                                 shuffle=False, num_workers=args.workers)

    u_trainloader = data.DataLoader(u_train_dataset, batch_size=args.train_batch,
                                  shuffle=True, num_workers=args.workers)
    
    print("Data loaders created !")
    # ############################### Model ###############################
    
    print("==> creating model ")
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
    model = torch.nn.DataParallel(model)
    model.cuda()
    print(model)
    
    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    
    # ############################### Optimizer and Loss ###############################
    # define criterion, optimizer for training
    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)

    # ############################### Resume ###############################
    title = args.dataset + "-" + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.save_dir = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.save_dir, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.save_dir, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train AUC.', 'Valid AUC.'])

    # evaluate with random initialization parameters
    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # save random initialization parameters
    save_checkpoint({'state_dict': model.state_dict()}, False, checkpoint=args.save_dir, filename='init.pth.tar')

    # ############################### Train and val ###############################
    all_result = {}
    all_result['train_acc'] = []
    all_result['val_acc'] = []
    all_result['test_acc'] = []
    best_model_epoch = -1
    
    fopen = open(args.save_dir + "auc_detail.txt", "w")
    
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        # train in one epoch
        train_loss, train_AUCs, train_acc = train(trainloader, u_trainloader, model, criterion, optimizer, epoch, use_cuda)

        # ######## acc on validation data each epoch ########
        dev_loss, dev_AUCs, dev_acc = test(devloader, model, criterion, epoch, use_cuda)
        
        # ######## acc on test data each epoch ########
        test_loss, test_AUCs, test_acc = test(testloader, model, criterion, 0, use_cuda)
        
        # append logger file
        logger.append([state['lr'], train_loss, dev_loss, train_acc, dev_acc])
        fopen.write("{} \t {} \t {}\n".format(epoch, test_AUCs, test_acc))
        fopen.flush()
        # save model after one epoch
        # Note: save all models after one epoch, to help find the best rewind
        is_best = dev_acc > best_acc
        if is_best:
            best_model_epoch = epoch
        best_acc = max(dev_acc, best_acc)
        
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': dev_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.save_dir, filename=str(epoch + 1)+'_checkpoint.pth.tar')
        
        # ############################### Plotting code ###############################
        
        all_result['train_acc'].append(train_acc.item())
        all_result['val_acc'].append(dev_acc.item())
        all_result['test_acc'].append(test_acc.item())
        
        fig, ax = plt.subplots(figsize=(10,6))
        plt.style.use('default')
        plt.plot(all_result['train_acc'], label='Train Acc.', color = "crimson")
        plt.plot(all_result['val_acc'], label='Val Acc.', color = "lightcoral")
        plt.plot(all_result['test_acc'], label='Test Acc.', color = "mediumseagreen")
        plt.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        plt.grid(axis = 'y')
        plt.savefig(os.path.join(args.save_dir, 'net_train.png'))
        plt.close()

    print('Best Validation Accuracy: {} \t||\t Epoch: {}'.format(best_acc, best_model_epoch))
    print(best_acc)

    # ################################### test ###################################
    print('Load best model ...')
    checkpoint = torch.load(os.path.join(args.save_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    test_loss, test_AUCs, test_acc = test(testloader, model, criterion, 0, use_cuda)
    fopen.write("------------" * 5)
    fopen.write("{} \t {} \t {}\n".format(epoch, test_AUCs, test_acc))
    fopen.flush()
    logger.append([state['lr'], -1, test_loss, -1, test_acc])
    print('test acc (best val acc)')
    print(test_acc)

    print('Load last model ...')
    checkpoint = torch.load(os.path.join(args.save_dir, str(epoch + 1)+'_checkpoint.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    test_loss, test_AUCs, test_acc = test(testloader, model, criterion, 0, use_cuda)
    logger.append([state['lr'], -1, test_loss, -1, test_acc])
    print('test acc (last epoch)')
    print(test_acc)

    logger.close()


def train(trainloader, u_trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()
    alpha = 0.7
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    auc = AverageMeter()
    
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    print(args)
    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    
    for batch_idx, ((image_name1, inputs1, targets1), (image_name2, inputs2, targets2)) in enumerate(zip(trainloader, u_trainloader)):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs1, targets1 = inputs1.cuda(), targets1.cuda()
            inputs2, targets2 = inputs2.cuda(), targets2.cuda()
        inputs1, targets1 = torch.autograd.Variable(inputs1), torch.autograd.Variable(targets1)
        inputs2, targets2 = torch.autograd.Variable(inputs2), torch.autograd.Variable(targets2)
        
        # compute output
        outputs1 = model(inputs1)
        outputs2 = model(inputs2)
        
        gt = torch.cat((gt, targets1.data), 0)
        pred = torch.cat((pred, outputs1.data), 0)
        
        
        loss1 = criterion(outputs1, targets1)
        loss2 = criterion(outputs2, targets2)
        
        loss = loss1 + (alpha * loss2)
        
        # measure accuracy and record loss
        AUCs, AUROC_avg = compute_AUCs(targets1.data, outputs1.data, targets1.shape[1])
        losses.update(loss1.item(), inputs1.size(0))
        auc.update(AUROC_avg, inputs1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | AUC: {top1: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=auc.avg
                    )
        bar.next()
    bar.finish()
    AUCs, AUROC_avg = compute_AUCs(gt, pred, targets1.shape[1])
    return (losses.avg, AUCs, AUROC_avg)


def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    auc = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    
    with torch.no_grad():
        for batch_idx, (image_name, inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)
            gt = torch.cat((gt, targets.data), 0)
            pred = torch.cat((pred, outputs.data), 0)
            
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            
            AUCs, AUROC_avg = compute_AUCs(targets.data, outputs.data, targets.shape[1])
            losses.update(loss.item(), inputs.size(0))
            auc.update(AUROC_avg, inputs.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | AUC: {top1: .4f} '.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=auc.avg
                        )
            bar.next()
    bar.finish()
    AUCs, AUROC_avg = compute_AUCs(gt, pred, targets.shape[1])
    return (losses.avg, AUCs, AUROC_avg)


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
