import sys
sys.path.append('/home/xuchengjun/ZXin/EARN')
from re import sub
import h5py
import torch
import cv2
import numpy as np
from torch._C import layout
from torch.utils.data import Dataset
from path import Path
from lib.utils import *
from config.config import cfg
from matplotlib import pyplot as plt
from IPython import embed
from sklearn.model_selection import train_test_split
import torch.nn as nn

class TrainDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        """
        初始化数据集
        """
        self.cfg = cfg
        # self.train_root_path = self.cfg.TRAIN.DATASET_PATH
        # self.label_root_path = self.cfg.TRAIN.LABEL_PATH
        self.cycles = self.cfg.TRAIN.CYCLES
        # self.data, self.labels = self.process_data(self.train_root_path, self.label_root_path)
        # self.data = self.data[0:150]
        # self.labels = self.labels[0:150]
        self.data = data
        self.labels = label
         
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        """
        data_path: 'train_data.h5' --> data (numpy)
        data: N C T J
        """
        data = self.data[index]

        data_numpy = np.asarray(data)   #　原始的形状是（54, joint_num * coor_num） --> (1, 54, 15, 3, 1)
        data_numpy = np.reshape(data_numpy,
                               (1,
                                data_numpy.shape[0],
                                self.cfg.DATASET.JOINT_NUM,
                                self.cfg.DATASET.COORD_NUM,
                                1))

        data_numpy = np.moveaxis(data_numpy, [1, 2, 3], [2, 3, 1])  # -->（1, 3, 54, 15, 1） 3代表的X,Y,Z三个维度
        self.N, self.C, self.T, self.J, self.M = data_numpy.shape
        img_data = self.convert_skeleton_to_image(data_numpy)   # 

        img_data = img_data.transpose((2, 0, 1)).astype(np.float32)
        img_data = torch.from_numpy(img_data).float()
        
        label = self.labels[index] -1 #label要从0开始  UTD-MAHD
        # label = self.labels[index]

        # no use
        # label = torch.tensor(int(label)).float()
        # data = torch.from_numpy(data).float()

        img_label = torch.tensor(int(label)).float()


        return img_data, img_label

        # return data, label
        
    def convert_skeleton_to_image(self, data_numpy):
        """
        data --> numpy
        将带有时序的骨骼数据转化成嵌入图像
        """
        data_numpy = np.squeeze(data_numpy, axis=0)
        data_max = np.max(data_numpy, (1, 2, 3))   # 这里会得到三个通道中最大的数值
        data_min = np.min(data_numpy, (1, 2, 3))   # 这里会得到三个通道中最小的数值
        img_data = np.zeros((
                              data_numpy.shape[1],
                              data_numpy.shape[2],
                              data_numpy.shape[0]   
                           ))

        # 你在作数据集的时候已经归一化过了
        # img_data[:, :, 0] = (data_max[0] - data_numpy[0, :, :, 0]) * (255 / (data_max[0] - data_min[0])) 
        # img_data[:, :, 1] = (data_max[1] - data_numpy[1, :, :, 0]) * (255 / (data_max[1] - data_min[1]))
        # img_data[:, :, 2] = (data_max[2] - data_numpy[2, :, :, 0]) * (255 / (data_max[2] - data_min[2]))

        img_data[:, :, 0] = data_numpy[0, :, :, 0] * 255
        img_data[:, :, 1] = data_numpy[1, :, :, 0] * 255
        img_data[:, :, 2] = data_numpy[2, :, :, 0] * 255

        #20220414  用不完整的姿态数据
        # img_data = np.pad(img_data, [(54-img_data.shape[0],0),(0,0),(0,0)],'constant',constant_values=0)
        
        img_data = cv2.resize(img_data, (self.cfg.TRAIN.IMAGE_SIZE, self.cfg.TRAIN.IMAGE_SIZE))
        # cv2.imwrite("new.jpg",img_data)

        return img_data
    
    def corvert_label_to_image(self, label):
        img_label = np.zeros((4,4))
        img_label[int(label), int(label)] = 1.0
        return img_label
    
    def process_data(self, train_root_path, label_root_path):
        """
        让所有的动作数据都满足需要的时间步
        """
        all_data = h5py.File(train_root_path, 'r')
        all_label = h5py.File(label_root_path, 'r')
        
        time_steps = 0
        # num_samples = len(all_data.keys())
        num_samples = 100
        labels = np.empty(num_samples)
        
        data_list = []
        for i in range(num_samples):  # 
            data_key = list(all_data.keys())[i]
            data_list.append(list(all_data[data_key]))
            time_steps_curr = len(all_data[data_key])
            if time_steps_curr > time_steps:  # 找到最大时间步
                time_steps = time_steps_curr
                
            labels[i] = all_label[list(all_label.keys())[i]][()]
            
        data = np.empty((num_samples, time_steps * self.cycles, self.cfg.DATASET.JOINT_NUM * self.cfg.DATASET.COORD_NUM))
        
        for j in range(num_samples):
            data_list_curr = np.tile(
                data_list[j], (int(np.ceil(time_steps / len(data_list[j]))), 1)   #这个是为了保持所有的数据的平等性，先得到最大的数据长度，然后小于的就直接数据平铺
            )
            
            for k in range(self.cycles):
                data[j, time_steps * k:time_steps * (k+1), :] = data_list_curr[0:time_steps]            

        return data, labels  
        
def process_data(cfg,train=False,val=False):
    """
    让所有的动作数据都满足需要的时间步
    """ 
    data_root_path = None
    label_root_path = None

    if train:
        data_root_path = cfg.TRAIN.DATASET_PATH 
        label_root_path = cfg.TRAIN.LABEL_PATH
    if val:
        data_root_path = cfg.TEST.DATASET_PATH 
        label_root_path = cfg.TEST.LABEL_PATH
    all_data = h5py.File(data_root_path, 'r')
    all_label = h5py.File(label_root_path, 'r')
    cycles = cfg.TRAIN.CYCLES

    time_steps = 0
    num_samples = len(all_data.keys())
    # num_samples = 200
    labels = np.empty(num_samples)
    
    data_list = []
    num = 1
    for i in range(num_samples):  # 
        data_key = list(all_data.keys())[i]
        data_list.append(list(all_data[data_key]))
        time_steps_curr = len(all_data[data_key])
        if time_steps_curr > time_steps:  # 找到最大时间步
            time_steps = time_steps_curr
        labels[i] = all_label[list(all_label.keys())[i]][()]
        # print(labels[i])
        print(f"working {num_samples}/{num}")
        num += 1
    print(time_steps)

    #平均帧数
    # data = np.empty((num_samples, time_steps * cycles, cfg.DATASET.JOINT_NUM * cfg.DATASET.COORD_NUM))
    # for j in range(num_samples):

    #     data_list_curr = np.tile(
    #         data_list[j], (int(np.ceil(time_steps / len(data_list[j]))), 1)   #这个是为了保持所有的数据的平等性，先得到最大的数据长度，然后小于的就直接数据平铺
    #     )
    #     for k in range(cycles):
    #         data[j, time_steps * k:time_steps * (k+1), :] = data_list_curr[0:time_steps]

    #这个是不平均帧数的
    data = []
    for j in range(num_samples):
        data.append(np.array(data_list[j]))

    return data, labels


def show_map(map, id):
    map = np.array(map) / 255

    plt.subplot(111)
    plt.imshow(map)
    plt.axis('off')
    # plt.savefig(f'img_{id}.jpg')
    plt.show()  

if __name__ == "__main__":
    data, labels = process_data(cfg,train=False,val=True)
    train_dataset = TrainDataset(data, labels)
    # embed()
    for i in range(len(train_dataset)):
        img_data, img_label = train_dataset[i]
        print(img_label)
        # embed()
        # show_map(img_data[:, :, 1], 1)
        show_map(img_data, 2)
        print('working ..')

    # rnn = nn.LSTM(input_size=45,hidden_size=45,num_layers=6,batch_first=True)
    # tanh = nn.Tanh()
    # input = torch.randn(2,54,45)
    # output,(hn,cn) = rnn(input)
    # embed()
