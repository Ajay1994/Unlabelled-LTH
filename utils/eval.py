from __future__ import print_function, absolute_import
import sys
from sklearn.metrics import roc_auc_score
import numpy as np
__all__ = ['accuracy', 'compute_AUCs']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def compute_AUCs(gt, pred, N_CLASSES):
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        except:
            continue
    AUROC_avg = np.array(AUROCs).mean()
    return AUROCs, AUROC_avg