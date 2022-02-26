from __future__ import print_function, division

#python prune.py --resume results/0/model_best.pth.tar --save_dir results/36 --percent 0.36

# pytorch imports
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
from dataloader import ChestXrayDataSet
from torchvision import datasets, models, transforms

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, compute_AUCs
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
use_cuda = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
print("Available GPU count:" + str(gpu_count))


from utils.misc import get_conv_zero_kernel
import argparse

# ############################### Parameters ###############################
parser = argparse.ArgumentParser(description='PyTorch Pruning')

# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_dir', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
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
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

# pruned ratio !!!
parser.add_argument('--percent', default=0.6, type=float)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

TRAIN_IMAGE_LIST = "train_val_list_labels.txt"
VAL_IMAGE_LIST = "test_list_labels.txt"
DATA_DIR = "/data/NIH_Xray/images/"


def main():

    os.makedirs(args.save_dir, exist_ok=True)
    # ############################### Dataset ###############################
    print('==> Preparing dataset ChestXray')
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


    
    print('Total image in train, ', len(train_dataset))
    print('Total image in valid, ', len(dev_dataset))
    print('Total image in test, ', len(test_dataset))
    

    trainloader = data.DataLoader(train_dataset, batch_size=args.train_batch,
                                  shuffle=True, num_workers=args.workers)

    devloader = data.DataLoader(dev_dataset, batch_size=args.train_batch,
                                shuffle=False, num_workers=args.workers)

    testloader = data.DataLoader(test_dataset, batch_size=args.test_batch,
                                 shuffle=False, num_workers=args.workers)
    
    
    
    # ############################### Model ###############################
    print("==> creating model ")
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
    
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # put model on GPU
    model = torch.nn.DataParallel(model)
    # ############################### Resume ###############################
    # Resume
    print("Resuming model ...")
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['state_dict'])
    
    
    model.cuda()
    
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
    # ############################### Test origin model ###############################
    print('\nEvaluation only')
    test_loss, test_AUCs, test_acc = test(testloader, model, criterion, 0, use_cuda)
    print("Before pruning ...\n{}".format(test_AUCs))
    print("--------------- Mean Auc : {} ---------------".format(test_acc))

    # -------------------------------------------------------------
    # ############################### pruning ###############################
    total = 0
    total_nonzero = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += m.weight.data.numel()
            mask = m.weight.data.abs().clone().gt(0).float().cuda()
            total_nonzero += torch.sum(mask)

    conv_weights = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            size = m.weight.data.numel()
            conv_weights[index:(index + size)] = m.weight.data.view(-1).abs().clone()
            index += size

    y, i = torch.sort(conv_weights)
    print(total, total_nonzero, total_nonzero/total)
    # thre_index = int(total * args.percent)
    # only care about the non zero weights
    # e.g: total = 100, total_nonzero = 80, percent = 0.2, thre_index = 36, that means keep 64
    thre_index = total - total_nonzero + int(total_nonzero * args.percent)
    thre = y[int(thre_index)]
    pruned = 0
    print('Pruning threshold: {}'.format(thre))
    zero_flag = False
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.numel() - torch.sum(mask)
            m.weight.data.mul_(mask)
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                  format(k, mask.numel(), int(torch.sum(mask))))
    print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned / total))
    # -------------------------------------------------------------

    # ############################### Test pruned model ###############################
    print('\nTesting')
    test_loss, test_AUCs, test_acc = test(testloader, model, criterion, 0, use_cuda)
    print("After pruning ...\n{}".format(test_AUCs))
    print("--------------- Mean Auc : {} ---------------".format(test_acc))

    state = {
        'state_dict': model.state_dict(),
        'best_loss': 0,
        'epoch': 0,
        'LR': 0
    }

    torch.save(state, args.save_dir + '/pruned.pth.tar')


    print("-------------------Kernel Ananlysis----------------")
    total_kernel, zero_kernel = get_conv_zero_kernel(model)
    
    with open(os.path.join(args.save_dir, 'prune.txt'), 'w') as f:
        f.write('After pruning: Test AUC:  %.3f\n' % (test_acc))
        f.write(
            'Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}\n'.format(total, pruned, pruned / total))
        f.write("Total Kernel: {} \t Zero Kernel: {}".format(total_kernel, zero_kernel))
        if zero_flag:
            f.write("There exists a layer with 0 parameters left.")
    return

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




if __name__ == '__main__':
    main()
