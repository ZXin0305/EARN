import torch.optim as optim
import torch
from torch.optim import lr_scheduler

def smooth_step(a, b, c, d, x):
    level_s=0.0001
    level_m=0.001
    level_n=0.00001
    level_r=0.000001
    if x <= a:
        return level_s
    if a < x <= b:
        return (((x-a)/(b-a))*(level_m-level_s)+level_s)
    if b < x <= c:
        return level_m
    if c < x <= d:
        return level_n
    if d < x:
        return level_r

def make_optimizer(cfg, model, epoch=0, sgd=False, smooth_lr=False, adam=True):
    optimizer = None
    if sgd:
        if smooth_lr:  # smooth_step(5,20,30,45, epoch)
            optimizer = optim.SGD(model.parameters(), lr=cfg.SOLVER.ADAM_BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)  # 10, 40, 100, 200
    if adam:
        optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.ADAM_BASE_LR , betas=(0.9, 0.999), eps=1e-08, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    # optimizer = optim.Adagrad(model.parameters(), lr=0.001)
        
    return optimizer

def make_lr_scheduler(cfg, optimizer, cosin=True, is_multiStep=False, is_sampleStep=False):
    scheduler = None
    base_lr = cfg.SOLVER.ADAM_BASE_LR
    total_epoch = cfg.TRAIN.END_EPOCH
    if cosin:
        lr_lambda = lambda epoch : (total_epoch - epoch) / total_epoch
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    elif is_multiStep:
        drop_after_epoch = [10,30,50,70,75,90,95]
        # drop_after_epoch = [i * 10 for i in range(1, int(cfg.TRAIN.END_EPOCH / 10) + 1)]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.1)
    elif is_sampleStep:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
    return scheduler


        