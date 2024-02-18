import torch
import torch.nn as nn
from typing import Iterable, Optional
from utils import log
from utils.log import get_grad_norm
from utils.loss import SalLoss
import math
from easydict import EasyDict


def train_one_epoch(model: nn.Module,
                    criterion: nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    cfg: EasyDict,
                    start_steps=None, update_freq=1, gamma=1.0):
    model.train()
    model.frozen_encoder()
    metric_logger = log.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', log.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', log.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for n_iter, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = n_iter // update_freq
        it = start_steps + step
        if cfg.DATA.USE_SOUND:
            imgs, audio, label = batch_data

            imgs = imgs.to(device, non_blocking=True)
            audio = audio.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            output, loss_va = model(imgs, audio)
            loss = criterion(output, label) + gamma * loss_va
        else:
            imgs, label = batch_data

            imgs = imgs.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            output, _ = model(imgs)
            loss = criterion(output, label)

        loss_value = loss.item()

        if math.isnan(loss_value):
            raise Exception("Loss is NaN.")
        metric_logger.update(loss=loss_value)
        metric_logger.update(kld=criterion.log['kl'].val)
        metric_logger.update(cc=criterion.log['cc'].val)
        metric_logger.update(sim=criterion.log['sim'].val)
        metric_logger.update(nss=criterion.log['nss'].val)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group['lr'])
            max_lr = max(max_lr, group['lr'])
        grad_norm = get_grad_norm(model.parameters())
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(model: nn.Module,
                         data_loader: Iterable,
                         device: torch.device,
                         cfg: EasyDict):
    criterion = SalLoss()

    metric_logger = log.MetricLogger(delimiter=" ")
    header = 'Val:'
    print_freq = 10

    model.eval()

    for n_iter, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if cfg.DATA.USE_SOUND:
            imgs, audio, label = batch_data

            imgs = imgs.to(device, non_blocking=True)
            audio = audio.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            output, _ = model(imgs, audio)
            loss = criterion(output, label)
        else:
            imgs, label = batch_data

            imgs = imgs.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            output, _ = model(imgs)
            loss = criterion(output, label)

        metric_logger.update(loss=loss.item())
        metric_logger.update(kld=criterion.log['kl'].val)
        metric_logger.update(cc=criterion.log['cc'].val)
        metric_logger.update(sim=criterion.log['sim'].val)

    metric_logger.synchronize_between_processes()
    print('* Kldiv {kldiv.global_avg:.3f} CC {cc.global_avg:.3f} SIM {sim.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(kldiv=metric_logger.kld, cc=metric_logger.cc, sim=metric_logger.sim, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
