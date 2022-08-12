import numpy as np
from IPython import embed

def pose_normalization(pred_3d_bodys):
    """[summary]
    original
    """
    origin_x = np.min(pred_3d_bodys[:,0])
    origin_y = np.min(pred_3d_bodys[:,1])
    origin_z = np.min(pred_3d_bodys[:,2])
 
    max_x = np.max(pred_3d_bodys[:,0])
    max_y = np.max(pred_3d_bodys[:,1])
    max_z = np.max(pred_3d_bodys[:,2])
    len_x = max_x - origin_x
    len_y = max_y - origin_y
    len_z = max_z - origin_z
        
    pred_3d_bodys[:,0] = np.round((pred_3d_bodys[:,0] - origin_x) / len_x, 3)
    pred_3d_bodys[:,1] = np.round((pred_3d_bodys[:,1] - origin_y) / len_y, 3)
    pred_3d_bodys[:,2] = np.round((pred_3d_bodys[:,2] - origin_z) / len_z, 3)
    if len_x == 0 or len_y == 0 or len_z ==0:
        return None
    
    return pred_3d_bodys

def vector_pose_normalization(pred_3d_bodys, root_idx):
    """
    用相对于根节点的姿态，这里也可以进行归一化，但是别只保留三位数了，感觉不太准确
    以m为单位
    """
    pred_3d_bodys = (pred_3d_bodys[root_idx] - pred_3d_bodys) / 100  #化成m
    origin_x = np.min(pred_3d_bodys[:,0])
    origin_y = np.min(pred_3d_bodys[:,1])
    origin_z = np.min(pred_3d_bodys[:,2])

    max_x = np.max(pred_3d_bodys[:,0])
    max_y = np.max(pred_3d_bodys[:,1])
    max_z = np.max(pred_3d_bodys[:,2])   

    len_x = max_x - origin_x
    len_y = max_y - origin_y
    len_z = max_z - origin_z

    pred_3d_bodys[:,0] = (pred_3d_bodys[:,0] - origin_x) / len_x
    pred_3d_bodys[:,1] = (pred_3d_bodys[:,1] - origin_y) / len_y
    pred_3d_bodys[:,2] = (pred_3d_bodys[:,2] - origin_z) / len_z

    return pred_3d_bodys

def change_pose(pred_3d_bodys):
    """[summary]

    Args:
        pred_3d_bodys ([type]): [description]
        not original 

    Returns:
        [type]: [description]
    """ 
    # pose_3d = []
    # for j in range(15):
    #     pose_3d.append(pred_3d_bodys[j][0])  # x
    #     pose_3d.append(pred_3d_bodys[j][1])  # y
    #     pose_3d.append(pred_3d_bodys[j][2])  # z
    pose_3d = pred_3d_bodys.flatten().tolist()
    return pose_3d  