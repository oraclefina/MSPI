import os
import time
import datetime
import random
import torch
import numpy as np
import argparse
import logging
import json
from config import cfg
from engine_train import train_one_epoch, validation_one_epoch
from avsp_dataloader import AudioVisualDataset
from torch.utils.data import ConcatDataset, DataLoader
from model.model_utils import AudioVisualSaliencyModel as SalModel
from utils.loss import SalLoss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="s1_mvitv2_epoch120_batch2_16_224_384")

    # training parameters
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--split", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int)

    parser.add_argument("--dataset", default='sound', type=str)
    parser.add_argument("--weights", type=str, default='')
    parser.add_argument("--log_dir", type=str, default='./training_logs')
    parser.add_argument("--save_ckpt", default=True, type=bool)
    parser.add_argument("--save_ckpt_freq", default=10, type=int)
    parser.add_argument("--gamma", default=1, type=float)

    args = parser.parse_args()

    # Fix Seed
    Seed = 2023
    random.seed(Seed)
    np.random.seed(Seed)
    torch.manual_seed(Seed)
    torch.cuda.manual_seed(Seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0")

    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    if os.path.exists(log_dir):
        log_dir = os.path.join(args.log_dir,
                               '{}_{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S'), np.random.randint(1, 10)))

    args.log_dir = log_dir

    # Checkpoints directory
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    # Data
    train_dataset_diem = AudioVisualDataset(data_root=cfg.DATA.ROOT, mode="train",
                                            dataset_name='DIEM', split=args.split, len_clip=cfg.DATA.NUM_FRAMES,
                                            use_sound=cfg.DATA.USE_SOUND, size=cfg.DATA.RESOLUTION)
    val_dataset_diem = AudioVisualDataset(data_root=cfg.DATA.ROOT, mode="test",
                                          dataset_name='DIEM', split=args.split, len_clip=cfg.DATA.NUM_FRAMES,
                                          use_sound=cfg.DATA.USE_SOUND, size=cfg.DATA.RESOLUTION)

    train_dataset_coutrout1 = AudioVisualDataset(data_root=cfg.DATA.ROOT, mode="train",
                                                 dataset_name='Coutrot_db1', split=args.split, len_clip=cfg.DATA.NUM_FRAMES,
                                                 use_sound=cfg.DATA.USE_SOUND, size=cfg.DATA.RESOLUTION)
    val_dataset_coutrout1 = AudioVisualDataset(data_root=cfg.DATA.ROOT, mode="test",
                                               dataset_name='Coutrot_db1', split=args.split, len_clip=cfg.DATA.NUM_FRAMES,
                                               use_sound=cfg.DATA.USE_SOUND, size=cfg.DATA.RESOLUTION)

    train_dataset_coutrout2 = AudioVisualDataset(data_root=cfg.DATA.ROOT, mode="train",
                                                 dataset_name='Coutrot_db2', split=args.split, len_clip=cfg.DATA.NUM_FRAMES,
                                                 use_sound=cfg.DATA.USE_SOUND, size=cfg.DATA.RESOLUTION)
    val_dataset_coutrout2 = AudioVisualDataset(data_root=cfg.DATA.ROOT, mode="test",
                                               dataset_name='Coutrot_db2', split=args.split, len_clip=cfg.DATA.NUM_FRAMES,
                                               use_sound=cfg.DATA.USE_SOUND, size=cfg.DATA.RESOLUTION)

    train_dataset_avad = AudioVisualDataset(data_root=cfg.DATA.ROOT, mode="train",
                                            dataset_name='AVAD', split=args.split, len_clip=cfg.DATA.NUM_FRAMES,
                                            use_sound=cfg.DATA.USE_SOUND, size=cfg.DATA.RESOLUTION)
    val_dataset_avad = AudioVisualDataset(data_root=cfg.DATA.ROOT, mode="test",
                                          dataset_name='AVAD', split=args.split, len_clip=cfg.DATA.NUM_FRAMES,
                                          use_sound=cfg.DATA.USE_SOUND, size=cfg.DATA.RESOLUTION)

    train_dataset_etmd = AudioVisualDataset(data_root=cfg.DATA.ROOT, mode="train",
                                            dataset_name='ETMD_av', split=args.split, len_clip=cfg.DATA.NUM_FRAMES,
                                            use_sound=cfg.DATA.USE_SOUND, size=cfg.DATA.RESOLUTION)
    val_dataset_etmd = AudioVisualDataset(data_root=cfg.DATA.ROOT, mode="test",
                                          dataset_name='ETMD_av', split=args.split, len_clip=cfg.DATA.NUM_FRAMES,
                                          use_sound=cfg.DATA.USE_SOUND, size=cfg.DATA.RESOLUTION)

    train_dataset_summe = AudioVisualDataset(data_root=cfg.DATA.ROOT, mode="train",
                                             dataset_name='SumMe', split=args.split, len_clip=cfg.DATA.NUM_FRAMES,
                                             use_sound=cfg.DATA.USE_SOUND, size=cfg.DATA.RESOLUTION)
    val_dataset_summe = AudioVisualDataset(data_root=cfg.DATA.ROOT, mode="test",
                                           dataset_name='SumMe', split=args.split, len_clip=cfg.DATA.NUM_FRAMES,
                                           use_sound=cfg.DATA.USE_SOUND, size=cfg.DATA.RESOLUTION)

    dataset_train = ConcatDataset([
        train_dataset_diem, train_dataset_coutrout1,
        train_dataset_coutrout2,
        train_dataset_avad, train_dataset_etmd,
        train_dataset_summe
    ])

    dataset_val = ConcatDataset([
        val_dataset_diem, val_dataset_coutrout1,
        val_dataset_coutrout2,
        val_dataset_avad, val_dataset_etmd,
        val_dataset_summe
    ])

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        num_workers=args.num_workers,
    )

    total_batch_size = cfg.TRAIN.BATCH_SIZE

    num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    if cfg.DATA.USE_SOUND:
        model = SalModel(cfg=cfg)
    else:
        from model.model_utils import VisualSaliencyModel as SalModel
        model = SalModel(cfg=cfg)
    model = model.to(device)

    # Frozen Encoder
    for name, param in model.named_parameters():
        if name.startswith('audnet'):
            param.requires_grad_(False)
        if name.startswith('image_encoder'):
            param.requires_grad_(False)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), cfg.SOLVER.LR,
                                  weight_decay=0)

    lr_schedule_values_by_epoch = [cfg.SOLVER.LR for _ in range(60)]
    LR = cfg.SOLVER.LR * 0.1
    for i in range(cfg.SOLVER.MAX_EPOCH - 60):
        lr_schedule_values_by_epoch.append(LR)
        if (i + 1) % 60 == 0:
            LR = LR * 0.1

    best_score = 100.0
    start_time = time.time()
    # Train
    for epoch in range(args.start_epoch, cfg.SOLVER.MAX_EPOCH):
        torch.cuda.empty_cache()
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule_values_by_epoch[epoch]

        train_stats = train_one_epoch(model, criterion=SalLoss(), data_loader=data_loader_train,
                                      optimizer=optimizer,
                                      device=device, epoch=epoch,
                                      start_steps=epoch * num_training_steps_per_epoch, gamma=args.gamma,cfg=cfg)

        if args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or (epoch + 1) == cfg.SOLVER.MAX_EPOCH:
                torch.cuda.empty_cache()
                model_save_path = os.path.join(checkpoint_dir, 'ckpt_{}.pth'.format(epoch + 1))
                torch.save(model.state_dict(), model_save_path)

        if epoch + 1 in {60, 80, 100, 120}:
            test_stats = validation_one_epoch(model, data_loader_val, device, cfg)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        with open(os.path.join(log_path, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
