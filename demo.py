from re import T
import torch
from config.config import cfg
from dataset.dataset import TrainDataset
from model.action import EARN
from model.lstm_model import LSTM_model
from model.vs_gcnn import VSGCNN
from torch.utils.data import DataLoader
from lib.solver import *
from lib.utils import *
from lib.judge import judge_action_label
import torch.nn as nn
import numpy as np
from path import Path
from IPython import embed
from dataset.dataset import process_data
from sklearn.metrics import confusion_matrix
from collections import OrderedDict

def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        # if c > 0.001:
        #     plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
        # if c == 0:
        #     plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
        if c > 0.5:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='white', fontsize=8, va='center', ha='center')
        elif c> 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=8, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes, rotation=30)
    plt.ylabel('Actual action')
    plt.xlabel('Predict action')
    
    # offset the tick
    tick_marks = np.array(range(len(classes)))
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()

# classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
#             '14,', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26']
# classes = ['ST', 'WL','WA','OA','RA']
classes = ['swipe left','swipe right', 'wave', 'clap', 'throw', 'arm cross', 'basketball shoot', 'draw x',
           'draw circle CW', 'draw circle CCW', 'draw triangle', 'bowling', 'boxing', 'baseball swing',
           'tennis swing', 'arm curl', 'tennis serve', 'push', 'knock', 'catch', 'pickup throw',
           'jog', 'walk', 'sit2stand', 'stand2sit', 'lunge', 'squat']

if __name__ == '__main__':
    model = EARN(depth=cfg.MODEL.DEPTH, num_classes=cfg.DATASET.NUM_CLASSES, widen_factor=cfg.MODEL.WIDEN_FACTOR, dropRate=cfg.MODEL.DROPOUT_RATE, nc=cfg.MODEL.INPUT_NC)
    model.to(cfg.TRAIN.DEVICE)

    # model = LSTM_model(45,45,6,6,5,5,54,45)
    # model.to(cfg.TRAIN.DEVICE)

    # model = VSGCNN(5,3,5,0.4)
    # model.to(cfg.TRAIN.DEVICE)

        # model_state_dict = ck['state_dict']
        # model_state_dict = ck


    if Path(cfg.MODEL.CKPT_PATH).exists():  #CKPT_PATH  BEST_PATH
        print(f'load checkpoint --> {cfg.MODEL.CKPT_PATH}')
        state_dict = torch.load(cfg.MODEL.CKPT_PATH)

        new_source_dict = OrderedDict()
        for k,v in state_dict.items(): #k:键名，v:对应的权值参数
            if k[0:7] == "module.":
                name = k[7:]
            else:
                name = k
            new_source_dict[name] = v

        model.load_state_dict(new_source_dict)
        
        # model.load_state_dict(state_dict['state_dict'])

        # model.load_state_dict(state_dict)
        print('loaded model state !') 
        
    # model.to('cuda:0')
    data_test, labels_test = process_data(cfg, val=True)
    val_dataset = TrainDataset(data_test, labels_test)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=0)
    
    
    model.eval()
    i = 0
    total = len(val_dataloader)
    correct = 0
    out_pre = []
    label_ = []
    for img_data, label in val_dataloader:
        with torch.no_grad():
            img_data = img_data.to(cfg.TRAIN.DEVICE)
            # output = model(img_data, apply_softmax=True)
            output = model(img_data)
            pre_label = int(output.argmax(1).detach().cpu())
          
            if pre_label == label:
                correct += 1

            out_pre.append(pre_label)
            label_.append(int(label))
            i+=1
            print(f"pre .. {len(val_dataloader)} / {i}")
    out_pre = np.array(out_pre)
    label_ = np.array(label_)
    # embed()
    cm = confusion_matrix(label_, out_pre)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    total = 0
    for i in range(27):
        total += cm_normalized[i][i]
    print(total / 27)
    plot_confusion_matrix(cm_normalized, 'confusion_matrix.png', title='confusion matrix')

    acc = correct / len(val_dataloader) 
    print(f'acc: {correct/total:0.4f}')
            
    