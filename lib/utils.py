import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import copy
from IPython import embed
from mayavi import mlab

bodys_eadges = [[0,1],[1,20],[20,2],[2,3],[0,16],[16,17],[17,18],[18,19],[0,12],[12,13],
                [13,14],[14,15],[20,8],[8,9],[9,10],[10,11],[11,23],[11,24],[20,4],
                [4,5],[5,6],[6,7],[7,22],[7,21]
               ]


def read_csv(path):
    try:
        data = pd.read_csv(path, header=0)
    except:
        print('dataset not exist')
        return 
    
    return data

def show_map(map, id):
    map = np.array(map) * 255

    plt.subplot(111)
    plt.imshow(map)
    plt.axis('off')
    plt.savefig(f'output_{id}.jpg')
    # plt.show()  

def augment_pose(pred_3d_bodys, scale=1, trans=0, angles=0, root_idx=2):
    pose_3d = copy.deepcopy(pred_3d_bodys)
    # human_num = pred_3d_bodys.shape[0]
    # joint_num = pred_3d_bodys.shape[1]

    joint_num = pred_3d_bodys.shape[0]

    # camera_theta = 45 * np.pi / 180

    # first
    trans_ = copy.deepcopy(pred_3d_bodys[root_idx, :3]) # --> (X, Y, Z) in camera coordinate system

    """
    20220418
    如果要加上不同的距离的话
    可以加上下面的代码，将trans_改变
    """
    # trans_ = [trans_[0] + trans * (trans_[0] / trans_[2]), trans_[1], trans_[2] + trans]

    pose_3d[:,0] -= trans_[0]
    pose_3d[:,1] -= trans_[1]
    pose_3d[:,2] -= trans_[2] 

    # 坐标转换到自身的坐标系中
    # 在自身坐标系中进行旋转
    # camera_rotMat = np.array([[1, 0, 0],
    #                           [0, np.cos(camera_theta), np.sin(camera_theta)],
    #                           [0, -np.sin(camera_theta), np.cos(camera_theta)]])
    # H_first = np.ones((4,4))
    # H_first[0:3, 0:3] = camera_rotMat.T
    # H_first[3, 0:3] = [0,0,0]
    # H_first[:3, 3] = [0,0,0]
    # pose_3d = pose_3d @ H_first
    
    theta = angles * np.pi / 180
    rotMat = np.array([[np.cos(theta), 0, -np.sin(theta)],
                        [0, 1, 0],
                        [np.sin(theta), 0, np.cos(theta)]])    #其实这里的旋转矩阵是绕y轴顺时针旋转的
    
    xyTrans = [trans * np.cos(theta), trans * np.sin(theta), 0]
    xyTrans = np.array(xyTrans)
    K = np.array([[scale, 0, 0, 0], [0, scale, 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]])
    H = np.ones((4, 4))
    H[0:3, 0:3] = rotMat.T    # 转置以后又变成了逆时针的
    H[3, :3] = xyTrans  # trans

    P = K @ H
    P[:3,3] = [0,0,0]

    data = np.ones((joint_num, 4))
    data[:,0:3] = pose_3d[:,0:3]
    data = data @ P 

    
    # 再转回到相机的坐标系中
    # 不用再转到之前的角度的相机坐标系了。。
       
    # rotMat_2 = np.array([[1, 0, 0],
    #                      [0, np.cos(-camera_theta), np.sin(-camera_theta)],
    #                      [0, -np.sin(-camera_theta), np.cos(-camera_theta)]])
    
    # H_ = np.ones((4,4))
    # H_[0:3, 0:3] = rotMat_2.T
    # H_[3, 0:3] = [0,0,0]
    # H_[:3, 3] = [0,0,0]
    # data = data @ H_  
    data[:,0] += trans_[0]
    data[:,1] += trans_[1]
    data[:,2] += trans_[2] 

    return data


def draw_3d_lines(mlab,p1,p2,color=(0,0,1)):
    xs = np.array([p1[0], p2[0]])
    ys = np.array([p1[1], p2[1]])
    zs = np.array([p1[2], p2[2]])
    mlab.plot3d(xs, ys, zs, [1, 2], tube_radius=0.01, color=color)

def draw_3d_sphere(mlab, point3d, color=(0,1,0)):
    mlab.points3d(
          np.array(point3d[0]), np.array(point3d[1]), np.array(point3d[2]),
          scale_factor=0.01, color=color
      )

def show_3d_results(pred_3d_poses):

    mlab.figure(1, bgcolor=(1,1,1), size=(960, 540))
    mlab.view(azimuth=180, elevation=0)

    for j in range(len(bodys_eadges)):
        p1 = pred_3d_poses[bodys_eadges[j][0]]
        p2 = pred_3d_poses[bodys_eadges[j][1]]
        draw_3d_lines(mlab, p1, p2)

    for j in range(25):
        draw_3d_sphere(mlab, pred_3d_poses[j])
    mlab.show()