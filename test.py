from IPython.terminal.embed import embed
import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
from path import Path
from lib.utils import *
import h5py


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

# def show_map(center_map):
#     center_map = np.array(center_map)

#     # center_map = center_map * 255
#     plt.subplot(111)
#     plt.imshow(center_map)
#     plt.axis('off')
#     plt.show()
# x = torch.randn(3,244,244)
# # conv = torch.nn.Conv1d(in_channels=244,out_channels=1,kernel_size=1,stride=1)
# conv = nn.AdaptiveAvgPool2d((None, 1))
# y = conv(x)
# y = y.expand(-1,244,244)
# y = y.detach().numpy()
# print(y.shape)
# show_map(x[0])
# show_map(y[0])


# useful_dir = ['0', '1', '2', '3']
root_path = '/media/xuchengjun/datasets/action_zx/0/train_data.csv'

# data_list = []
# label = []
# sub_dirs = Path(root_path).dirs()
# for sub_dir in sub_dirs:
#     if sub_dir.basename() in useful_dir:
#         for file in sub_dir.files():
#             data_list.append(file)
#             label.append(int(sub_dir.basename()))
    
# data = read_csv(root_path)    

# f = h5py.File('/media/xuchengjun/datasets/action_zx/test.h5', 'w')
# f['test'] = data

# xx = np.zeros((4,4))
# print(xx)

# label_position = [(4 + 1) * i for i in range(4) ]
# print(label_position)

# label_vector = np.zeros(4 * 4)
# print(label_vector)
# embed()

# xx = torch.Tensor([[[1,2,3],[4,5,6],[7,8,9]]])
# yy = xx.view((-1, 9))
# print(yy)

# xx = np.sum(100 >= [2000])
# print(xx)


# def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):

#     plt.figure(figsize=(12, 8), dpi=100)
#     np.set_printoptions(precision=2)

#     # 在混淆矩阵中每格的概率值
#     ind_array = np.arange(len(classes))
#     x, y = np.meshgrid(ind_array, ind_array)
#     for x_val, y_val in zip(x.flatten(), y.flatten()):
#         c = cm[y_val][x_val]
#         if c > 0.001:
#             plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
#     plt.title(title)
#     plt.colorbar()
#     xlocations = np.array(range(len(classes)))
#     plt.xticks(xlocations, classes, rotation=90)
#     plt.yticks(xlocations, classes)
#     plt.ylabel('Actual label')
#     plt.xlabel('Predict label')
    
#     # offset the tick
#     tick_marks = np.array(range(len(classes))) + 0.5
#     plt.gca().set_xticks(tick_marks, minor=True)
#     plt.gca().set_yticks(tick_marks, minor=True)
#     plt.gca().xaxis.set_ticks_position('none')
#     plt.gca().yaxis.set_ticks_position('none')
#     plt.grid(True, which='minor', linestyle='-')
#     plt.gcf().subplots_adjust(bottom=0.15)
    
#     # show confusion matrix
#     plt.savefig(savename, format='png')
#     plt.show()

# classes = ['0', '1', '2', '3', '4']

# # random_numbers = np.random.randint(5, size=100)  # 6个类别，随机生成50个样本
# # y_true = random_numbers.copy()  # 样本实际标签
# # random_numbers[:10] = np.random.randint(5, size=10)  # 将前10个样本的值进行随机更改
# # y_pred = random_numbers  # 样本预测标

# random_numbers = np.array([0,1,2,3,4,0])  
# y_true = random_numbers.copy()  # 样本实际标签
# random_numbers_ = np.array([0,1,0,3,4,2]) 
# y_pred = random_numbers_  # 样本预测标签



# cm = confusion_matrix(y_true, y_pred)
# cm_normalized = cm.astype('flo)at') / cm.sum(axis=1)[:, np.newaxis]
# plot_confusion_matrix(cm_normalized, 'confusion_matrix.png', title='confusion matrix')


drop_after_epoch = [i * 10 for i in range(1, int(150 / 10) + 1)]
print(drop_after_epoch)


