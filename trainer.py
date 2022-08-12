
from re import T
from IPython.terminal.embed import embed
import torch
from torch.optim import optimizer 
from config.config import cfg
from dataset.dataset import TrainDataset
from model.action import EARN
# from model.action_v2 import EARN
from model.lstm_model import LSTM_model
from model.vs_gcnn import VSGCNN
from torch.utils.data import DataLoader
from lib.solver import *
from lib.utils import *
import torch.nn as nn
import numpy as np
from path import Path
from lib.judge import judge_action_label
from dataset.dataset import process_data
import random
import os
from collections import OrderedDict

class Trainer():
    def __init__(self, cfg, train_dataloader, val_dataloader) -> None:
        self.cfg = cfg
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = nn.CrossEntropyLoss()
        
        
    def train(self, model, current_epoch, optimizer, scheduler, iter, sgd, adam):
        train_loss = []
        iter_ = iter
        for (img_data, img_label) in (self.train_dataloader):
            iter_ += 1
            optimizer.zero_grad()
            img_data = img_data.type(torch.FloatTensor)
            img_data = img_data.to(self.cfg.TRAIN.DEVICE)
            img_label = img_label.long()
            img_label = img_label.to(self.cfg.TRAIN.DEVICE)
            # img_label = (self.cfg.DATASET.NUM_CLASSES * img_label + img_label).to(self.cfg.TRAIN.DEVICE)

            output = model(img_data)

            # pre_label = output.argmax(1)
            loss = self.loss_fn(output, img_label.long()) * self.cfg.TRAIN.LOSS_FACTOR
            loss.backward(retain_graph=True)
            optimizer.step()
                
            train_loss.append(loss.data.item())
        
            print('Epoch: {} process: {}/{} Loss: {} LR: {}'.format(current_epoch, iter_, 
                                                                    len(self.train_dataloader), 
                                                                    np.mean(np.array(train_loss)),
                                                                    optimizer.param_groups[0]["lr"]))
            
            if current_epoch != 0 and current_epoch % self.cfg.MODEL.SAVE_PERIOD == 0:
                torch.save(model.module.state_dict(), Path(self.cfg.MODEL.SAVE_PATH) / f'utd_sv2_{current_epoch}.pth')
            
            if iter_ != 0 and iter_ % self.cfg.MODEL.CKPT_PERIOD == 0:
                if adam:
                    ck = {
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iter': iter_,
                        'current_epoch':current_epoch
                    }
                else:
                    ck = {
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter': iter_,
                        'current_epoch':current_epoch
                    }                    
                torch.save(ck, Path(self.cfg.MODEL.CKPT_PATH))
        if sgd:
            current_lr = smooth_step(10, 40, 100, 160, current_epoch)
            self.update_lr(optimizer, current_lr)
        if adam:
            scheduler.step()
    
    def train_lstm(self,model,current_epoch,optimizer,scheduler,iter,sgd,adam):
        train_loss = []
        iter_ = iter
        for (pose_sequence, label) in (self.train_dataloader):
            iter_ += 1
            optimizer.zero_grad()
            pose_sequence.to(self.cfg.TRAIN.DEVICE)
            label = label.long()
            label = label.to(self.cfg.TRAIN.DEVICE)
            output = model(pose_sequence)
            loss = self.loss_fn(output, label)
            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss.append(loss.data.item())

            print('Epoch: {} process: {}/{} Loss: {} LR: {}'.format(current_epoch, iter_, 
                                                                    len(self.train_dataloader), 
                                                                    np.mean(np.array(train_loss)),
                                                                    optimizer.param_groups[0]["lr"]))            

            if current_epoch != 0 and current_epoch % self.cfg.MODEL.SAVE_PERIOD == 0:
                torch.save(model.module.state_dict(), Path(self.cfg.MODEL.SAVE_PATH) / f'lstm_{current_epoch}.pth')
            
            if iter_ != 0 and iter_ % self.cfg.MODEL.CKPT_PERIOD == 0:
                if adam:
                    ck = {
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iter': iter_,
                        'current_epoch':current_epoch
                    }
                else:
                    ck = {
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter': iter_,
                        'current_epoch':current_epoch
                    }                    
                torch.save(ck, Path(self.cfg.MODEL.CKPT_PATH))
        if sgd:
            current_lr = smooth_step(5,20,30,45, current_epoch)
            self.update_lr(optimizer, current_lr)
        if adam:
            scheduler.step()

    def test(self, model, vsgcnn=False):
        model.eval()
        correct = 0
        for img_data, label in self.val_dataloader:
            with torch.no_grad():
                img_data = img_data.to(self.cfg.TRAIN.DEVICE)
                output = model(img_data)
                pre_label = output.argmax(1).detach().cpu()
                if vsgcnn:
                    label = label * self.cfg.DATASET.NUM_CLASSES + label
                if pre_label == label:
                    correct += 1
        acc = correct / len(self.val_dataloader) 
        return acc          

                
    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
    def cal_loss(self, pre, gt):
        loss = 0.
        batch_size = gt.shape[0]
        for i in range(batch_size):
            loss += torch.sum(torch.abs(pre - gt))
        return loss / batch_size

    
    def train_vsgcnn(self, model,current_epoch,optimizer,scheduler,iter,sgd,adam):
        train_loss = []
        iter_ = iter
        for (img_data, img_label) in (self.train_dataloader):
            iter_+=1
            optimizer.zero_grad()
            img_data = img_data.type(torch.FloatTensor)
            img_data = img_data.to(self.cfg.TRAIN.DEVICE)
            img_label = img_label.to(self.cfg.TRAIN.DEVICE)
            # img_label = (self.cfg.DATASET.NUM_CLASSES * img_label + img_label).to(self.cfg.TRAIN.DEVICE)

            output = model(img_data)
            loss = self.loss_fn(output,img_label.long())
            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss.append(loss.data.item())

            print('Epoch: {} process: {}/{} Loss: {} LR: {}'.format(current_epoch, iter_, 
                                                                    len(self.train_dataloader), 
                                                                    np.mean(np.array(train_loss)),
                                                                    optimizer.param_groups[0]["lr"]))
            
            if current_epoch != 0 and current_epoch % self.cfg.MODEL.SAVE_PERIOD == 0:
                torch.save(model.module.state_dict(), Path(self.cfg.MODEL.SAVE_PATH) / f'vsgcnn_myData{current_epoch}.pth')
            
            if iter_ != 0 and iter_ % self.cfg.MODEL.CKPT_PERIOD == 0:
                if adam:
                    ck = {
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iter': iter_,
                        'current_epoch':current_epoch
                    }
                else:
                    ck = {
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter': iter_,
                        'current_epoch':current_epoch
                    }                    
                torch.save(ck, Path(self.cfg.MODEL.CKPT_PATH))  

        if sgd:
            current_lr = smooth_step(10, 40, 100, 160, current_epoch)
            self.update_lr(optimizer, current_lr)
        if adam:
            scheduler.step()           


    
def main(model, trainer, cfg):
    # init params
    start_epoch = cfg.TRAIN.START_EPOCH
    end_epoch = cfg.TRAIN.END_EPOCH
    iter = 0
    
    # init solver
    sgd = False
    adam = True
    scheduler = None
    optimizer = None
    optimizer = make_optimizer(cfg, model, start_epoch, sgd=sgd, adam=adam)
    if adam:
        scheduler = make_lr_scheduler(cfg, optimizer)
    
    # load ckpt
    if Path(cfg.MODEL.CKPT_PATH).exists():
        print(f'load checkpoint --> {cfg.MODEL.CKPT_PATH}')
        ck = torch.load(cfg.MODEL.CKPT_PATH)
        
        # model_state_dict = ck['state_dict']
        # model_state_dict = ck

        # new_source_dict = OrderedDict()
        # for k,v in model_state_dict.items(): #k:键名，v:对应的权值参数
        #     if k[0:7] == "module":
        #         name = k[7:]
        #     else:
        #         name = k
        #     new_source_dict[name] = v
        # model.load_state_dict(new_source_dict)

        model.load_state_dict(ck['state_dict'])
        print('loaded model state !')
        
        optimizer_state_dict = ck['optimizer']
        optimizer.load_state_dict(optimizer_state_dict)
        print('loaded optimizer state !')

        if adam:
            scheduler_state_dict = ck['scheduler']
            scheduler.load_state_dict(scheduler_state_dict)
            print('loaded scheduler state !')
        
        start_epoch = ck['current_epoch'] + 1
        print(f'loaded current start_epoch ! {start_epoch}')
        
        iter = ck['iter']
        print(f'loaded current iteration ! {iter}')

    if cfg.TRAIN.DP_MODE:
        print('using DP Mode ..')
        model = torch.nn.DataParallel(model,device_ids = cfg.TRAIN.GPU_IDS)
        
    model.train()
    best_acc = 0
    best_epoch = 0
    for current_epoch in range(start_epoch, end_epoch + 1):
        trainer.train(model, current_epoch, optimizer, scheduler, iter, sgd, adam)
        # trainer.train_lstm(model, current_epoch, optimizer, scheduler, iter, sgd, adam)
        # trainer.train_vsgcnn(model, current_epoch, optimizer, scheduler, iter, sgd, adam)
        acc = trainer.test(model)
        print(f"当前的epoch： {current_epoch}, 准确率为: {acc:0.4f}")
        if acc >= best_acc:
            best_acc = acc
            best_epoch = current_epoch
            torch.save(model.module.state_dict(), Path(cfg.MODEL.BEST_PATH))
        iter = 0
        model.train()
    print(f"best acc: {best_acc} best_epoch: {best_epoch}")

def set_seed(seed=18):
    # seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

if __name__ == '__main__':
    # set_seed()
    model = EARN(depth=cfg.MODEL.DEPTH, num_classes=cfg.DATASET.NUM_CLASSES, widen_factor=cfg.MODEL.WIDEN_FACTOR, dropRate=cfg.MODEL.DROPOUT_RATE, nc=cfg.MODEL.INPUT_NC, 
                 out_dim=32, flatten_size=1)
    model.to(cfg.TRAIN.DEVICE)
    print('train the EARN ..')

    # model = LSTM_model(45,45,6,10,0,5,54,45)
    # model.to(cfg.TRAIN.DEVICE)
    # print('train the LSTM ..')

    # model = VSGCNN(5,3,5,0.2)
    # model.to(cfg.TRAIN.DEVICE)
    # print('train the VSGCNN ..')
    print('created a new model ..')
    # data_train, data_test, labels_train, labels_test = process_data(cfg)
    # train_dataset = TrainDataset(data_train, labels_train)
    # val_dataset = TrainDataset(data_test, labels_test)
    train_dataloader = None
    val_dataloader = None
    data_train, labels_train = process_data(cfg,train=True)
    data_test, labels_test = process_data(cfg,val=True)
    train_dataset = TrainDataset(data_train, labels_train)
    val_dataset = TrainDataset(data_test, labels_test)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKER,drop_last=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=0)
    # embed()
    trainer = Trainer(cfg, train_dataloader, val_dataloader)
    main(model, trainer, cfg)