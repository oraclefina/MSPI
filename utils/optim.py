import math
import time
import os
import json
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict
from torch.utils.data._utils.collate import default_collate
from pathlib import Path
import subprocess
import torch
import torch.distributed as dist
from torch._six import inf
import random
from einops import rearrange
import torch.nn.functional as F

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))
    # schedule = np.ones_like(schedule) * 1e-4

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args, epoch, model, optimizer):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    checkpoint_path = os.path.join(output_dir, 'checkpoint-%s.pt' % epoch_name)
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(to_save, checkpoint_path)

def auto_load_model(args, model, optimizer):
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1

            print('With optim & sched!')

def interpolate_embeddings(feature_size, aud_size, model_state, T=2, interpolation_mode='bicubic'):
    vis_pos_embedding = model_state["aud_vis_sync_block.vis_pos_token"]
    aud_pos_embedding = model_state["aud_vis_sync_block.aud_pos_token"]

    new_vis_pos_embedding = rearrange(vis_pos_embedding, 'n (t h w) d -> n (d t) h w', t=T, h=7, w=7)
    new_vis_pos_embedding = F.interpolate(new_vis_pos_embedding, size=feature_size, mode=interpolation_mode)
    new_vis_pos_embedding = rearrange(new_vis_pos_embedding, 'n (d t) h w -> n (t h w) d',t=T)

    new_aud_pos_embedding = rearrange(aud_pos_embedding, 'n (t f) c -> n c t f', t=9)
    new_aud_pos_embedding = F.interpolate(new_aud_pos_embedding, size=aud_size, mode=interpolation_mode)
    new_aud_pos_embedding = rearrange(new_aud_pos_embedding, 'n c t f -> n (t f) c')


    model_state["aud_vis_sync_block.vis_pos_token"] = new_vis_pos_embedding
    model_state["aud_vis_sync_block.aud_pos_token"] = new_aud_pos_embedding

    return model_state


