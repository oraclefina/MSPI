import torch.nn as nn
from .compute_saliency_metrics import *
from timm.utils import AverageMeter
import torch.nn.functional as F

class SalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log = {
            'kl': AverageMeter(),
            'cc': AverageMeter(),
            'sim': AverageMeter(),
            'nss': AverageMeter(),
            'loss': AverageMeter(),
        }

    def reset_records(self):
        self.log = {
            'kl': AverageMeter(),
            'cc': AverageMeter(),
            'sim': AverageMeter(),
            'nss': AverageMeter(),
            'loss': AverageMeter(),
        }

    def forward(self, inputs, targets, fixations=None, targets2=None):
        kl_loss = kldiv(inputs.exp(), targets)  
        cc_loss = cc(inputs.exp(), targets)
        sim_loss = similarity(inputs.exp(), targets)
        # bhattacharyya_loss = bhattacharyya_distance(inputs.exp(), targets2)

        if fixations is None:
            loss = kl_loss-cc_loss 
            
            self.log['kl'].update(kl_loss.item())
            self.log['cc'].update(cc_loss.item())
            self.log['sim'].update(sim_loss.item())
            self.log['loss'].update(loss.item())
        else:
            nss_loss = nss(inputs.exp(), fixations)
            loss = kl_loss - cc_loss - 0.1 * nss_loss

            self.log['kl'].update(kl_loss.item())
            self.log['cc'].update(cc_loss.item())
            self.log['sim'].update(sim_loss.item())
            self.log['nss'].update(nss_loss.item())
            self.log['loss'].update(loss.item())


        return loss
