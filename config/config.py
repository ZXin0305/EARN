from easydict import EasyDict as edit
from path import Path

class Config:
    TRAIN = edit()
    # /media/xuchengjun/datasets/action_zx/data_train.h5    
    # /media/xuchengjun/datasets/UTD-MAD/data_train.h5  /media/xuchengjun/datasets/UTD-MAD/cs2/data_train.h5
    # /media/xuchengjun/datasets/action_zx/data_train_tiny.h5
    # /media/xuchengjun/datasets/MSRAction3D/data_train.h5
    # /media/xuchengjun/datasets/action_zx/data_train_wo.h5
    # /media/xuchengjun/datasets/UTD-MHAD-MV/new_data/aug_train_data.h5'

    # /media/xuchengjun/datasets/MSRAction3D/cs2/data_test.h5
    # /media/xuchengjun/datasets/MSRAction3D/cs2/label_test.h5
    
    # /media/xuchengjun/datasets/action_zx/label_train.h5  
    # /media/xuchengjun/datasets/UTD-MAD/label_train.h5 
    # /media/xuchengjun/datasets/action_zx/label_train_tiny.h5
    # /media/xuchengjun/datasets/MSRAction3D/label_train.h5
    # /media/xuchengjun/datasets/action_zx/label_train_wo.h5
    # TRAIN.DATASET_PATH = '/media/xuchengjun/datasets/MSRAction3D/cs2/data_train.h5'
    # TRAIN.LABEL_PATH = '/media/xuchengjun/datasets/MSRAction3D/cs2/label_train.h5'


    TRAIN.DATASET_PATH = '/media/xuchengjun/disk/datasets/UTD-MAD/cs2/data_train.h5'
    TRAIN.LABEL_PATH = '/media/xuchengjun/disk/datasets/UTD-MAD/cs2/label_train.h5'
    TRAIN.CYCLES = 1
    TRAIN.DEVICE = 'cuda'
    TRAIN.START_EPOCH = 1
    TRAIN.END_EPOCH = 150
    TRAIN.BATCH_SIZE = 32  # 32
    TRAIN.NUM_WORKER = 0
    TRAIN.DP_MODE = True
    TRAIN.GPU_IDS = [0,1,2]
    TRAIN.NUM_GPU = len(TRAIN.GPU_IDS)
    TRAIN.LOSS_FACTOR = 1
    TRAIN.IMAGE_SIZE = 244

    TEST = edit()
    # /media/xuchengjun/datasets/action_zx/data_test.h5
    # /media/xuchengjun/datasets/UTD-MAD/data_test.h5   /media/xuchengjun/datasets/UTD-MAD/cs2/data_test.h5
    # /media/xuchengjun/datasets/action_zx/label_test.h5
    # /media/xuchengjun/datasets/UTD-MAD/label_test.h5        cross_subject
    # /media/xuchengjun/datasets/action_zx/data_test_tiny.h5
    # /media/xuchengjun/datasets/action_zx/label_test_tiny.h5
    # /media/xuchengjun/datasets/MSRAction3D/data_test.h5
    # /media/xuchengjun/datasets/MSRAction3D/label_test.h5
    # /media/xuchengjun/datasets/action_zx/data_test_wo.h5
    # /media/xuchengjun/datasets/action_zx/label_test_wo.h5
    # /media/xuchengjun/datasets/UTD-MHAD-MV/new_data/aug_test_data.h5'
    # /media/xuchengjun/datasets/UTD-MHAD-MV/new_data/aug_test_label.h5'
    # TEST.DATASET_PATH = '/media/xuchengjun/datasets/MSRAction3D/cs2/data_test.h5'
    # TEST.LABEL_PATH = '/media/xuchengjun/datasets/MSRAction3D/cs2/label_test.h5'
    
    # /media/xuchengjun/datasets/MSRAction3D/cs2/data_test.h5
    # /media/xuchengjun/datasets/MSRAction3D/cs2/label_test.h5
    TEST.DATASET_PATH = '/media/xuchengjun/disk/datasets/UTD-MAD/cs2/data_test.h5'
    TEST.LABEL_PATH = '/media/xuchengjun/disk/datasets/UTD-MAD/cs2/label_test.h5'
    
    DATASET = edit()
    DATASET.JOINT_NUM = 20  #15 20
    DATASET.COORD_NUM = 3
    DATASET.NUM_CLASSES = 27  #5 27 20
    
    MODEL = edit()
    MODEL.DEPTH = 4 * 6 + 4  #改第一个数
    MODEL.WIDEN_FACTOR = 4
    MODEL.DROPOUT_RATE = 0.0
    MODEL.INPUT_NC = 3  # 这个是输入的图像的通道数
    MODEL.SAVE_PERIOD = 1  #每多少个epoch进行保存
    MODEL.SAVE_PATH = '/home/xuchengjun/ZXin/EARN/trained_model'
    MODEL.CKPT_PERIOD = 1000000 # 每300个iter进行保存
    MODEL.CKPT_PATH = '/home/xuchengjun/ZXin/EARN/ckpt/train.pth'   # /home/xuchengjun/ZXin/EARN/ckpt/train.pth  /home/xuchengjun/ZXin/EARN/trained_model/other_20.pth 
                                                                                 # /home/xuchengjun/ZXin/EARN/trained_model/vsgcnn_23.pth
    MODEL.BEST_PATH = '/home/xuchengjun/ZXin/EARN/ckpt/best.pth'
    
    # 优化器
    SOLVER = edit()
    SOLVER.MOMENTUM = 0.9
    SOLVER.WEIGHT_DECAY = 1e-5
    SOLVER.ADAM_BASE_LR = 0.001
    
     
config = Config()
cfg = config      #cfg是创建好的类对象
